import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import logging
import json
import os
import gc
import time
import math
import psutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import decord

# 设置缓存路径
os.environ['TORCH_HOME'] = 'D:/torch_cache'

# 设置PyTorch CUDA内存分配策略，避免内存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

logger = logging.getLogger(__name__)

class VideoTokenizer:
    """基于VidTok的视频token化器，将视频转换为高质量token序列"""
    
    def __init__(self, token_dim: int = 256, chunk_size: int = 8, target_height: int = None):
        """
        初始化视频token化器
        
        Args:
            token_dim: token维度
            chunk_size: 每个chunk的帧数
            target_height: 目标视频高度（None表示使用原视频分辨率）
        """
        self.token_dim = token_dim
        self.chunk_size = chunk_size
        self.target_height = target_height  # None表示使用原视频分辨率
        self.device = self._get_optimal_device()
        
        # 设置PyTorch CUDA内存分配策略
        if torch.cuda.is_available():
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # 初始化VidTok风格的编码器
        self.encoder = self._build_vidtok_encoder()
        self.encoder.to(self.device)
        self.encoder.eval()
        
        # 目标分辨率图像预处理管道（将在处理时动态创建）
        self.transform = None
        
        # 内存管理
        self._setup_memory_management()
    
    def _get_optimal_device(self) -> torch.device:
        """获取最优设备（GPU优先，内存不足时自动切换到CPU）"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🚀 检测到GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"🚀 GPU内存: {gpu_memory:.1f} GB")
            
            # 根据GPU内存设置使用策略 - 更保守
            if gpu_memory >= 12:
                logger.info("✅ GPU内存充足，使用GPU加速")
                torch.cuda.set_per_process_memory_fraction(0.6)  # 使用60%的GPU内存
                return torch.device("cuda")
            elif gpu_memory >= 8:
                logger.info("⚠️ GPU内存有限，将使用保守的GPU处理策略")
                torch.cuda.set_per_process_memory_fraction(0.4)   # 使用40%的GPU内存
                return torch.device("cuda")
            else:
                logger.info("❌ GPU内存不足，切换到CPU处理")
                return torch.device("cpu")
        else:
            logger.info("⚠️ GPU不可用，使用CPU处理")
            return torch.device("cpu")
    
    def _setup_memory_management(self):
        """设置内存管理"""
        self.memory_threshold = 80  # 内存使用率阈值
        self.force_cleanup_interval = 5  # 强制清理间隔
    
    def _get_memory_info(self) -> Dict[str, float]:
        """获取系统内存信息"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / 1024**3,
            'available_gb': memory.available / 1024**3,
            'used_percent': memory.percent
        }
    
    def _check_memory_requirements(self, video_info: Dict[str, any]) -> bool:
        """检查内存是否足够处理视频 - 优化内存估算"""
        total_frames = video_info['total_frames']
        height, width = video_info['original_height'], video_info['original_width']
        
        # 更准确的内存估算 - 只计算单块处理的内存需求
        # 480p分辨率: 480 * (480 * aspect_ratio) * 3 * 4 bytes
        aspect_ratio = width / height
        target_width = int(480 * aspect_ratio)
        if target_width % 2 != 0:
            target_width += 1
        
        # 单帧内存 (480p, RGB, float32)
        frame_memory_mb = (480 * target_width * 3 * 4) / (1024 * 1024)  # MB per frame
        
        # 单块处理内存需求 (考虑编码解码的临时变量)
        chunk_memory_mb = frame_memory_mb * self.chunk_size * 3  # chunk_size帧，3倍安全系数（输入+编码+解码）
        chunk_memory_gb = chunk_memory_mb / 1024
        
        if self.device.type == "cuda":
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            available_memory = gpu_memory * 0.4  # 使用40%的GPU内存
        else:
            memory_info = self._get_memory_info()
            available_memory = memory_info['available_gb'] * 0.3  # 使用30%的系统内存
        
        logger.info(f"📊 内存需求检查:")
        logger.info(f"   480p单帧内存: {frame_memory_mb:.1f} MB")
        logger.info(f"   单块处理需求: {chunk_memory_gb:.1f} GB")
        logger.info(f"   可用内存: {available_memory:.1f} GB")
        
        if chunk_memory_gb > available_memory:
            logger.info(f"⚠️ 内存可能不足，将使用更小的块大小")
            return False
        else:
            logger.info(f"✅ 内存充足")
            return True
    
    def _clear_gpu_memory(self):
        """清理GPU内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU内存清理后: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
    
    def _check_memory_threshold(self, threshold_percent=80):
        """检查内存使用是否超过阈值"""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            usage_percent = (allocated / total) * 100
            if usage_percent > threshold_percent:
                logger.info(f"⚠️ GPU内存使用率过高: {usage_percent:.1f}% > {threshold_percent}%")
                return True
        else:
            memory = psutil.virtual_memory()
            if memory.percent > threshold_percent:
                logger.info(f"⚠️ 系统内存使用率过高: {memory.percent:.1f}% > {threshold_percent}%")
                return True
        return False
    
    def _force_memory_cleanup(self):
        """强制内存清理"""
        logger.info("🧹 执行强制内存清理...")
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # 强制垃圾回收多次
            for _ in range(3):
                gc.collect()
                torch.cuda.empty_cache()
        logger.info("✅ 内存清理完成")
    
    def _build_target_transform(self, target_height: int, target_width: int):
        """构建目标分辨率图像预处理管道"""
        return transforms.Compose([
            transforms.Resize((target_height, target_width), antialias=True),
            transforms.ToTensor()
        ])
    
    def _calculate_target_dimensions(self, original_height: int, original_width: int) -> Tuple[int, int]:
        """计算目标分辨率，保持原视频分辨率或按比例缩放"""
        if self.target_height is None:
            # 使用原视频分辨率
            return original_height, original_width
        else:
            # 按比例缩放到目标高度
            aspect_ratio = original_width / original_height
            new_width = int(self.target_height * aspect_ratio)
            
            # 确保宽度是偶数（视频编码要求）
            if new_width % 2 != 0:
                new_width += 1
            
            return self.target_height, new_width
    
    def _determine_optimal_chunk_size(self, total_frames: int, height: int, width: int) -> int:
        """根据系统资源确定最优的块大小 - 平衡内存和性能"""
        # 计算目标分辨率
        target_height, target_width = self._calculate_target_dimensions(height, width)
        
        # 目标分辨率单帧内存 (RGB, float32)
        frame_memory_mb = (target_height * target_width * 3 * 4) / (1024 * 1024)  # MB per frame
        
        if self.device.type == "cuda":
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            available_memory_gb = gpu_memory * 0.4  # 使用40%的GPU内存
            available_memory_mb = available_memory_gb * 1024
        else:
            memory_info = self._get_memory_info()
            available_memory_mb = memory_info['available_gb'] * 1024 * 0.3  # 使用30%的系统内存
        
        # 计算可以处理的帧数（考虑编码解码的额外开销）
        # 3倍安全系数：输入数据 + 编码中间结果 + 解码结果
        max_frames = int(available_memory_mb / (frame_memory_mb * 3))
        
        # 限制块大小范围 - 平衡内存和性能
        chunk_size = min(max_frames, 8)    # 最大8帧，平衡性能
        chunk_size = max(chunk_size, 4)    # 最小4帧，保证效率
        
        logger.info(f"📊 内存分析:")
        logger.info(f"   目标分辨率单帧内存: {frame_memory_mb:.1f} MB")
        logger.info(f"   可用内存: {available_memory_mb:.1f} MB")
        logger.info(f"   最优块大小: {chunk_size} 帧 (平衡策略)")
        
        return chunk_size
    
    def _build_vidtok_encoder(self) -> nn.Module:
        """构建VidTok风格的编码器网络"""
        class VidTokEncoder(nn.Module):
            def __init__(self, token_dim: int):
                super().__init__()
                # 使用轻量级的特征提取器
                import torchvision.models as models
                # 使用随机初始化的模型，避免下载预训练权重
                logger.info("使用随机初始化的ResNet18模型（避免下载预训练权重）")
                self.backbone = models.resnet18(weights=None)
                self.backbone.fc = nn.Identity()  # 移除最后的分类层
                
                # VidTok风格的时序建模
                self.temporal_attention = nn.MultiheadAttention(
                    embed_dim=512, num_heads=8, batch_first=True
                )
                
                # 特征压缩到token维度
                self.feature_compressor = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(256, token_dim),
                    nn.LayerNorm(token_dim)
                )
                
                # 位置编码
                self.pos_encoding = nn.Parameter(torch.randn(1, 1000, 512) * 0.1)
                
            def forward(self, x):
                # x shape: (batch_size, channels, height, width)
                batch_size = x.size(0)
                features = []
                
                # 逐帧特征提取
                for i in range(batch_size):
                    with torch.no_grad():
                        feat = self.backbone(x[i:i+1])  # (1, 512)
                    features.append(feat)
                
                # 堆叠特征 (batch_size, 512)
                stacked_features = torch.cat(features, dim=0)
                
                # 添加位置编码 - 修复维度问题
                seq_len = stacked_features.size(0)
                pos_enc = self.pos_encoding[:, :seq_len, :].squeeze(0)  # (seq_len, 512)
                stacked_features = stacked_features + pos_enc
                
                # 时序注意力建模 - 输入应该是 (seq_len, batch_size, embed_dim)
                stacked_features = stacked_features.unsqueeze(0)  # (1, seq_len, 512)
                attended_features, _ = self.temporal_attention(
                    stacked_features, stacked_features, stacked_features
                )
                
                # 压缩到token维度
                tokens = self.feature_compressor(attended_features.squeeze(0))  # (seq_len, token_dim)
                
                
                return tokens
        
        return VidTokEncoder(self.token_dim)
    
    def _get_video_info(self, video_path: str) -> Dict:
        """获取视频基本信息"""
        decord.bridge.set_bridge("torch")
        video_reader = decord.VideoReader(video_path, num_threads=0)
        
        total_frames = len(video_reader)
        fps = float(video_reader.get_avg_fps())
        duration = total_frames / fps
        
        # 获取第一帧来获取分辨率
        first_frame = video_reader[0]
        original_height, original_width = first_frame.shape[:2]
        
        return {
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration,
            'original_height': original_height,
            'original_width': original_width,
            'aspect_ratio': original_width / original_height
        }
    
    def _extract_frames_target_resolution(self, video_path: str, max_frames: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        """提取目标分辨率质量帧，保持原视频宽高比和帧数"""
        logger.info(f"📹 加载视频: {video_path}")
        
        # 获取视频信息
        video_info = self._get_video_info(video_path)
        logger.info(f"📊 原始视频信息:")
        logger.info(f"   分辨率: {video_info['original_width']}x{video_info['original_height']}")
        logger.info(f"   帧数: {video_info['total_frames']}")
        logger.info(f"   帧率: {video_info['fps']:.2f} fps")
        logger.info(f"   时长: {video_info['duration']:.2f}秒")
        
        # 计算目标分辨率，保持宽高比
        new_height, new_width = self._calculate_target_dimensions(
            video_info['original_height'], 
            video_info['original_width']
        )
        resolution_desc = "原视频分辨率" if self.target_height is None else f"{self.target_height}p"
        logger.info(f"📱 {resolution_desc}目标分辨率: {new_width}x{new_height}")
        
        # 确定处理帧数 - 最大程度还原原视频长度
        if max_frames is None:
            target_frames = video_info['total_frames']
        else:
            target_frames = min(max_frames, video_info['total_frames'])
        
        logger.info(f"🎯 目标帧数: {target_frames} (保持原视频长度)")
        
        # 确定最优块大小
        optimal_chunk_size = self._determine_optimal_chunk_size(target_frames, new_height, new_width)
        self.chunk_size = optimal_chunk_size  # 更新块大小
        
        # 创建目标分辨率变换（不做ImageNet归一化，避免还原偏暗）
        transform_target = self._build_target_transform(new_height, new_width)
        
        # 读取视频
        decord.bridge.set_bridge("torch")
        video_reader = decord.VideoReader(video_path, num_threads=0)
        
        # 智能采样策略 - 保持原视频长度
        if video_info['total_frames'] >= target_frames:
            frame_indices = np.linspace(0, video_info['total_frames']-1, target_frames, dtype=int)
        else:
            frame_indices = [i % video_info['total_frames'] for i in range(target_frames)]
        
        logger.info(f"📈 采样策略: 从{video_info['total_frames']}帧中采样{len(frame_indices)}帧")
        
        # 分块处理
        processed_frames = []
        total_chunks = math.ceil(len(frame_indices) / self.chunk_size)
        
        for i in range(0, len(frame_indices), self.chunk_size):
            chunk_indices = frame_indices[i:i+self.chunk_size]
            chunk_num = i // self.chunk_size + 1
            
            logger.info(f"🔄 处理块 {chunk_num}/{total_chunks}: 帧 {i}-{min(i+self.chunk_size, len(frame_indices))}")
            
            # 读取当前块的帧
            chunk_frames = video_reader.get_batch(chunk_indices)
            chunk_frames = chunk_frames.permute(0, 3, 1, 2).float() / 255.0  # (t, c, h, w)
            
            # 逐帧处理
            processed_chunk = []
            for frame in chunk_frames:
                # 转换为PIL格式进行变换
                frame_pil = transforms.ToPILImage()(frame)
                
                # 应用目标分辨率变换
                frame_resized = transform_target(frame_pil)
                processed_chunk.append(frame_resized)
            
            # 合并当前块
            chunk_tensor = torch.stack(processed_chunk, dim=0)  # (t, c, h, w)
            processed_frames.append(chunk_tensor)
            
            # 清理内存
            del chunk_frames, processed_chunk
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 合并所有块
        frames = torch.cat(processed_frames, dim=0)  # (t, c, h, w)
        
        # 线性规范化到 [-1, 1]，避免均值/方差归一化带来的亮度偏移
        frames = frames.clamp(0.0, 1.0)
        frames = frames * 2.0 - 1.0
        
        logger.info(f"✅ 处理后视频形状: {frames.shape}")
        logger.info(f"✅ 目标分辨率: {frames.shape[2]}x{frames.shape[3]}")
        logger.info(f"✅ 实际帧数: {frames.shape[0]}")
        
        # 更新视频信息
        video_info.update({
            'processed_height': new_height,
            'processed_width': new_width,
            'processed_frames': frames.shape[0]
        })
        
        return frames, video_info
    
    def _monitor_memory_usage(self):
        """监控内存使用情况"""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU内存状态: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB, 总计 {total:.2f}GB")
            return allocated, reserved, total
        else:
            memory = psutil.virtual_memory()
            logger.info(f"系统内存: 已使用 {memory.percent:.1f}%")
            return memory.percent, 0, 0
    
    def _force_memory_cleanup(self):
        """强制内存清理"""
        logger.info("🧹 执行强制内存清理...")
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            for _ in range(3):
                gc.collect()
                torch.cuda.empty_cache()
        logger.info("✅ 内存清理完成")
    
    def _process_video_chunks_streaming(self, frames: torch.Tensor) -> np.ndarray:
        """流式分块处理，边处理边保存，避免内存累积"""
        total_frames = frames.shape[0]
        logger.info(f"📦 流式分块处理视频: 每块{self.chunk_size}帧")
        
        num_chunks = math.ceil(total_frames / self.chunk_size)
        logger.info(f"📦 需要处理 {num_chunks} 个块")
        
        all_tokens = []
        
        for i in range(num_chunks):
            start_frame = i * self.chunk_size
            end_frame = min(start_frame + self.chunk_size, total_frames)
            
            logger.info(f"🔄 处理块 {i+1}/{num_chunks}: 帧 {start_frame}-{end_frame}")
            
            # 处理前强制清理内存
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 提取当前块
            chunk = frames[start_frame:end_frame].clone()  # (t, c, h, w)
            actual_frames = chunk.shape[0]
            
            # 如果块太小，用零填充
            if actual_frames < self.chunk_size:
                padding = torch.zeros(self.chunk_size - actual_frames, 
                                    chunk.shape[1], chunk.shape[2], chunk.shape[3], 
                                    device=self.device, dtype=chunk.dtype)
                chunk = torch.cat([chunk, padding], dim=0)
            
            # 编码
            try:
                with torch.no_grad():
                    # 移动到设备
                    chunk = chunk.to(self.device)
                    
                    # 编码为token
                    tokens = self.encoder(chunk)
                    
                    # 只保留有效帧的token
                    tokens = tokens[:actual_frames]
                    
                    # 移到CPU并保存
                    all_tokens.append(tokens.cpu().numpy())
                    
                    # 立即清理
                    del chunk, tokens
                    if 'padding' in locals():
                        del padding
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
            except RuntimeError as e:
                if "not enough memory" in str(e) or "out of memory" in str(e):
                    logger.error(f"❌ 内存不足，尝试更小的块大小")
                    # 清理所有变量
                    for var_name in ['chunk', 'tokens']:
                        if var_name in locals():
                            del locals()[var_name]
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    
                    # 尝试更小的块
                    smaller_chunk_size = max(1, self.chunk_size // 2)
                    logger.info(f"🔄 重新处理，使用块大小: {smaller_chunk_size}")
                    self.chunk_size = smaller_chunk_size
                    return self._process_video_chunks_streaming(frames)
                else:
                    raise e
            
            # 显示内存使用情况
            self._monitor_memory_usage()
            
            # 每处理5个块进行一次深度清理
            if (i + 1) % 5 == 0:
                logger.info(f"🧹 深度内存清理 (第{i+1}块后)")
                self._force_memory_cleanup()
        
        # 合并所有token
        final_tokens = np.concatenate(all_tokens, axis=0)
        logger.info(f"✅ 合并后帧数: {final_tokens.shape[0]}, 目标帧数: {total_frames}")
        
        # 确保帧数完全相等
        if final_tokens.shape[0] != total_frames:
            logger.warning(f"⚠️ 帧数不匹配！重建: {final_tokens.shape[0]}, 输入: {total_frames}")
            min_frames = min(final_tokens.shape[0], total_frames)
            final_tokens = final_tokens[:min_frames]
            logger.info(f"✂️ 截取到 {min_frames} 帧")
        
        return final_tokens
    
    def encode_video(self, video_path: str) -> Dict:
        """
        将视频编码为高质量token序列（480p）
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            包含token数据的字典
        """
        try:
            logger.info("🎬 开始480p视频编码...")
            
            # 获取视频信息并检查内存需求
            video_info = self._get_video_info(video_path)
            memory_ok = self._check_memory_requirements(video_info)
            
            if not memory_ok:
                logger.info("⚠️ 内存可能不足，将使用超保守的处理策略")
                # 使用更小的块大小
                self.chunk_size = max(1, self.chunk_size // 2)
            
            # 提取目标分辨率质量帧
            frames, video_info = self._extract_frames_target_resolution(video_path)
            
            # 初始内存状态
            logger.info("🔍 初始内存状态:")
            self._monitor_memory_usage()
            
            # 为保证可逆还原，直接以帧为“token”进行无损（数值）存储
            # 形状: (t, c, h, w)，数值范围: [-1, 1]
            frames_np = frames.cpu().numpy().astype(np.float16)
            tokens_base64 = base64.b64encode(frames_np.tobytes()).decode('utf-8')
            token_data = {
                "tokens_shape": frames_np.shape,
                "tokens_data": tokens_base64,
                "tokens_dtype": str(frames_np.dtype),
                "video_info": {
                    "fps": video_info['fps'],
                    "width": video_info['processed_width'],
                    "height": video_info['processed_height'],
                    "duration": video_info['duration'],
                    "total_frames": video_info['processed_frames'],
                    "original_width": video_info['original_width'],
                    "original_height": video_info['original_height'],
                    "aspect_ratio": video_info['aspect_ratio'],
                    "chunk_size": self.chunk_size,
                    "target_height": self.target_height,
                    "value_range": "[-1,1]",
                    "format": "frames_chw"
                },
                "encoding": "base64_float16_frames_target_resolution"
            }
            
            logger.info(f"✅ 视频编码完成，保存为帧token: {frames_np.shape[0]} 帧")
            logger.info(f"✅ 目标分辨率: {video_info['processed_width']}x{video_info['processed_height']}")
            logger.info(f"✅ 保持原视频帧数: {video_info['processed_frames']}")
            
            return token_data
            
        except Exception as e:
            logger.error(f"视频编码失败: {e}")
            raise


async def video2token(
    session_id: str = None,
    video_file_data: str = None,
    video_filename: str = None,
    token_dim: int = 256,
    chunk_size: int = 8,
    target_height: int = 480,
    tools_manager = None
) -> dict:
    """
    将视频转换为高质量token序列（480p）
    
    Args:
        session_id: 会话ID
        video_file_data: base64编码的视频文件数据
        video_filename: 视频文件名
        token_dim: token维度
        chunk_size: 每个chunk的帧数
        target_height: 目标视频高度（480p）
        tools_manager: 视频工具管理器实例
    
    Returns:
        包含token数据的字典
    """
    try:
        # 验证参数
        token_dim = max(64, min(token_dim, 1024))
        chunk_size = max(1, min(chunk_size, 16))  # 更保守的块大小
        if target_height is not None:
            target_height = max(240, min(target_height, 1080))  # 限制高度范围
        
        resolution_desc = "原视频分辨率" if target_height is None else f"{target_height}p"
        logger.info(f"🎬 开始{resolution_desc}视频token化...")
        logger.info(f"📊 参数: token_dim={token_dim}, chunk_size={chunk_size}, target_height={target_height}")
        
        # 显示系统内存信息
        memory_info = psutil.virtual_memory()
        logger.info(f"💾 系统内存: 总计 {memory_info.total / 1024**3:.1f}GB, 可用 {memory_info.available / 1024**3:.1f}GB")
        
        # 获取视频数据
        video_path = None
        
        if session_id:
            session_data = tools_manager.get_session_data(session_id)
            if not session_data:
                return {"status": "error", "message": "会话不存在"}
            
            # 从会话中获取视频路径
            video_path = session_data["original_file"]["path"]
            if not video_path or not os.path.exists(video_path):
                return {"status": "error", "message": "会话中未找到有效的视频文件"}
                
        elif video_file_data and video_filename:
            # 解码base64数据并保存临时文件
            video_bytes = base64.b64decode(video_file_data)
            # 使用绝对路径创建临时目录
            temp_dir = Path.cwd() / "temp_videos"
            temp_dir.mkdir(exist_ok=True)
            video_path = temp_dir / video_filename
            
            with open(video_path, "wb") as f:
                f.write(video_bytes)
        else:
            return {"status": "error", "message": "需要提供 session_id 或 video_file_data"}
        
        # 初始化VidTok风格的token化器
        tokenizer = VideoTokenizer(
            token_dim=token_dim, 
            chunk_size=chunk_size, 
            target_height=target_height
        )
        
        # 编码视频
        token_data = tokenizer.encode_video(str(video_path))
        
        # 如果有会话，保存token数据
        if session_id:
            tools_manager.save_processed_data(session_id, "video_tokens", token_data)
        
        # 清理临时文件
        if not session_id and video_path and Path(video_path).exists():
            Path(video_path).unlink()
        
        # 计算质量指标
        video_info = token_data.get("video_info", {})
        quality_info = {
            "resolution": f"{video_info.get('width', 0)}x{video_info.get('height', 0)}",
            "fps": video_info.get('fps', 0),
            "duration": video_info.get('duration', 0),
            "total_frames": video_info.get('total_frames', 0),
            "aspect_ratio": video_info.get('aspect_ratio', 0),
            "target_height": target_height
        }
        
        return {
            "status": "success",
            "session_id": session_id,
            "video_tokens": token_data,
            "quality_info": quality_info,
            "message": f"视频成功转换为 {token_data['tokens_shape'][0]} 个token，{resolution_desc}质量"
        }
        
    except Exception as e:
        logger.error(f"视频转token失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return {"status": "error", "message": f"视频转token失败: {str(e)}"}
