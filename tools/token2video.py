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

class VideoDecoder:
    """基于VidTok的视频解码器，将token序列转换回高质量视频"""
    
    def __init__(self, token_dim: int = 256, target_height: int = None):
        """
        初始化视频解码器
        
        Args:
            token_dim: token维度
            target_height: 目标视频高度（None表示使用原视频分辨率）
        """
        self.token_dim = token_dim
        self.target_height = target_height  # None表示使用原视频分辨率
        self.device = self._get_optimal_device()
        
        # 设置PyTorch CUDA内存分配策略
        if torch.cuda.is_available():
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # 初始化VidTok风格的解码器网络
        self.decoder = self._build_vidtok_decoder()
        self.decoder.to(self.device)
        self.decoder.eval()
        
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
    
    def _determine_optimal_chunk_size(self, total_tokens: int, token_dim: int) -> int:
        """根据系统资源确定最优的块大小 - 平衡内存和性能"""
        # 估算token处理的内存需求
        token_memory_mb = (total_tokens * token_dim * 4) / (1024 * 1024)  # MB for tokens
        
        if self.device.type == "cuda":
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            available_memory_gb = gpu_memory * 0.4  # 使用40%的GPU内存
            available_memory_mb = available_memory_gb * 1024
        else:
            memory_info = self._get_memory_info()
            available_memory_mb = memory_info['available_gb'] * 1024 * 0.3  # 使用30%的系统内存
        
        # 计算可以处理的token数（考虑解码的额外开销）
        # 3倍安全系数：输入token + 解码中间结果 + 输出帧
        max_tokens = int(available_memory_mb / (token_memory_mb * 3 / total_tokens))
        
        # 限制块大小范围 - 平衡内存和性能
        chunk_size = min(max_tokens, 8)    # 最大8个token，平衡性能
        chunk_size = max(chunk_size, 4)    # 最小4个token，保证效率
        
        logger.info(f"📊 内存分析:")
        logger.info(f"   Token内存需求: {token_memory_mb:.1f} MB")
        logger.info(f"   可用内存: {available_memory_mb:.1f} MB")
        logger.info(f"   最优块大小: {chunk_size} tokens (平衡策略)")
        
        return chunk_size
    
    def _build_vidtok_decoder(self) -> nn.Module:
        """构建VidTok风格的解码器网络"""
        class VidTokDecoder(nn.Module):
            def __init__(self, token_dim: int, target_height: int = None):
                super().__init__()
                self.target_height = target_height
                
                # 从token维度恢复到特征维度
                self.token_expander = nn.Sequential(
                    nn.Linear(token_dim, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(256, 512),
                    nn.LayerNorm(512)
                )
                
                # 时序注意力解码
                self.temporal_attention = nn.MultiheadAttention(
                    embed_dim=512, num_heads=8, batch_first=True
                )
                
                # 特征到空间映射
                self.feature_to_spatial = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 512),
                    nn.ReLU(inplace=True)
                )
                
                # 计算目标尺寸
                target_size = max(224, target_height) if target_height else 224
                
                # 使用转置卷积进行上采样到目标分辨率
                self.upsample_layers = nn.Sequential(
                    # 从 1x1 到 7x7
                    nn.ConvTranspose2d(512, 256, kernel_size=7, stride=1, padding=0),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    
                    # 从 7x7 到 14x14
                    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    
                    # 从 14x14 到 28x28
                    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    
                    # 从 28x28 到 56x56
                    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    
                    # 从 56x56 到 112x112
                    nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    
                    # 从 112x112 到 224x224
                    nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(8),
                    nn.ReLU(inplace=True),
                    
                    # 从 224x224 到目标尺寸
                    nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
                    nn.Tanh()  # 输出范围[-1, 1]
                )
                
                # 自适应调整到目标尺寸（如果指定了target_height）
                if target_height:
                    self.final_resize = nn.AdaptiveAvgPool2d((target_height, target_height))
                else:
                    self.final_resize = None
                
            def forward(self, tokens):
                # tokens shape: (batch_size, token_dim)
                batch_size = tokens.size(0)
                
                # 扩展token到特征维度
                features = self.token_expander(tokens)  # (batch_size, 512)
                
                # 添加位置编码
                seq_len = features.size(0)
                pos_encoding = torch.randn(1, 1, 512, device=tokens.device) * 0.1
                features = features.unsqueeze(1) + pos_encoding
                
                # 时序注意力解码
                attended_features, _ = self.temporal_attention(
                    features, features, features
                )
                
                # 压缩到空间特征
                spatial_features = self.feature_to_spatial(attended_features.squeeze(1))  # (batch_size, 512)
                
                # 重塑为空间特征图 (batch_size, 512, 1, 1)
                spatial_features = spatial_features.unsqueeze(-1).unsqueeze(-1)
                
                # 上采样生成图像
                frames = self.upsample_layers(spatial_features)
                
                # 调整到目标尺寸（如果指定了）
                if self.final_resize is not None:
                    frames = self.final_resize(frames)
                
                return frames
        
        return VidTokDecoder(self.token_dim, self.target_height)
    
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
    
    def _decode_tokens_to_frames_streaming(self, tokens: np.ndarray, chunk_size: int = 8) -> List[np.ndarray]:
        """流式解码token为帧序列，支持内存管理"""
        total_tokens = tokens.shape[0]
        token_dim = tokens.shape[1] if len(tokens.shape) > 1 else 256
        
        # 确定最优块大小
        optimal_chunk_size = self._determine_optimal_chunk_size(total_tokens, token_dim)
        chunk_size = min(chunk_size, optimal_chunk_size)  # 使用更保守的块大小
        
        logger.info(f"📦 流式解码token: 每块{chunk_size}个token")
        
        num_chunks = math.ceil(total_tokens / chunk_size)
        logger.info(f"📦 需要处理 {num_chunks} 个块")
        
        all_frames = []
        
        for i in range(num_chunks):
            start_token = i * chunk_size
            end_token = min(start_token + chunk_size, total_tokens)
            
            logger.info(f"token2video:🔄 处理块 {i+1}/{num_chunks}: 帧 {start_token}-{end_token}")
            
            # 处理前强制清理内存
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 提取当前块的token
            chunk_tokens = tokens[start_token:end_token]
            actual_tokens = chunk_tokens.shape[0]
            
            # 如果块太小，用零填充
            if actual_tokens < chunk_size:
                padding = np.zeros((chunk_size - actual_tokens, chunk_tokens.shape[1]), dtype=chunk_tokens.dtype)
                chunk_tokens = np.vstack([chunk_tokens, padding])
                logger.info(f"   填充token: {actual_tokens} -> {chunk_size}")
            
            # 解码
            try:
                with torch.no_grad():
                    # 转换为tensor并移动到设备
                    tokens_tensor = torch.from_numpy(chunk_tokens.copy()).float().to(self.device)
                    
                    # 解码为帧
                    frames_tensor = self.decoder(tokens_tensor)
                    
                    # 只保留有效token对应的帧
                    frames_tensor = frames_tensor[:actual_tokens]
                    
                    # 后处理每一帧
                    processed_chunk = []
                    for j in range(frames_tensor.shape[0]):
                        frame_tensor = frames_tensor[j]  # (3, H, W)
                        
                        # 反归一化（线性），避免额外偏移
                        frame_tensor = torch.clamp((frame_tensor + 1.0) * 0.5, 0.0, 1.0)
                        
                        # 转换为PIL图像
                        pil_image = transforms.ToPILImage()(frame_tensor)
                        
                        # 转换为numpy数组 (H, W, C)
                        frame_np = np.array(pil_image)
                        processed_chunk.append(frame_np)
                    
                    all_frames.extend(processed_chunk)
                    
                    # 立即清理
                    del tokens_tensor, frames_tensor, processed_chunk
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
                    for var_name in ['tokens_tensor', 'frames_tensor', 'processed_chunk']:
                        if var_name in locals():
                            del locals()[var_name]
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    
                    # 尝试更小的块
                    smaller_chunk_size = max(1, chunk_size // 2)
                    logger.info(f"🔄 重新解码，使用块大小: {smaller_chunk_size}")
                    return self._decode_tokens_to_frames_streaming(tokens, smaller_chunk_size)
                else:
                    raise e
            
            # 显示内存使用情况
            self._monitor_memory_usage()
            
            # 每处理5个块进行一次深度清理
            if (i + 1) % 5 == 0:
                logger.info(f"🧹 深度内存清理 (第{i+1}块后)")
                self._force_memory_cleanup()
        
        logger.info(f"✅ 解码完成，生成 {len(all_frames)} 帧")
        return all_frames
    
    def _frames_to_video_target_resolution(self, frames: List[np.ndarray], output_path: str, 
                                          fps: float, original_width: int, original_height: int) -> bool:
        """将帧序列合成为目标分辨率高质量视频"""
        try:
            resolution_desc = "原视频分辨率" if self.target_height is None else f"{self.target_height}p"
            logger.info(f"🎬 合成{resolution_desc}视频: {output_path}")
            
            if self.target_height is None:
                # 使用原视频分辨率
                target_height = original_height
                target_width = original_width
            else:
                # 计算目标分辨率，保持宽高比
                aspect_ratio = original_width / original_height
                target_width = int(self.target_height * aspect_ratio)
                
                # 确保宽度是偶数（视频编码要求）
                if target_width % 2 != 0:
                    target_width += 1
                
                target_height = self.target_height
            
            logger.info(f"📱 目标分辨率: {target_width}x{target_height}")
            
            # 使用更高质量的视频编码器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_path, 
                fourcc, 
                fps, 
                (target_width, target_height),
                isColor=True
            )
            
            if not out.isOpened():
                logger.error("无法创建视频写入器")
                return False
            
            logger.info(f"📊 开始写入 {len(frames)} 帧...")
            
            for i, frame in enumerate(frames):
                # 调整帧尺寸到目标分辨率
                resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
                
                # 转换RGB到BGR
                bgr_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
                
                # 写入帧
                out.write(bgr_frame)
                
                # 显示进度
                if (i + 1) % 100 == 0 or i == len(frames) - 1:
                    progress = (i + 1) / len(frames) * 100
                    logger.info(f"📈 进度: {progress:.1f}% ({i+1}/{len(frames)})")
            
            out.release()
            
            # 验证输出文件
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                logger.info(f"✅ 视频合成成功，文件大小: {file_size:.2f} MB")
                return True
            else:
                logger.error("输出视频文件不存在")
                return False
            
        except Exception as e:
            logger.error(f"视频合成失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            return False
    
    def decode_video(self, token_data: Dict, output_path: str) -> Dict:
        """
        将token序列解码为高质量视频（480p）
        
        Args:
            token_data: 包含token数据的字典
            output_path: 输出视频路径
            
        Returns:
            包含解码结果的字典
        """
        try:
            logger.info("🎬 开始480p视频解码...")
            
            # 解码token/帧数据（优先支持无损帧格式）
            tokens_bytes = base64.b64decode(token_data["tokens_data"])
            dtype_str = token_data.get("tokens_dtype", "float32")
            np_dtype = np.float16 if "16" in str(dtype_str) else np.float32
            frames_or_tokens = np.frombuffer(tokens_bytes, dtype=np_dtype)
            frames_or_tokens = frames_or_tokens.reshape(token_data["tokens_shape"])  # (t, c, h, w) 或 (t, d)
            logger.info(f"📊 载入数据形状: {frames_or_tokens.shape}, dtype={np_dtype}")
            
            # 获取视频信息
            video_info = token_data.get("video_info", {})
            fps = video_info.get("fps", 30.0)
            original_width = video_info.get("original_width", 224)
            original_height = video_info.get("original_height", 224)
            aspect_ratio = video_info.get("aspect_ratio", 1.0)
            
            logger.info(f"📊 原始视频信息:")
            logger.info(f"   分辨率: {original_width}x{original_height}")
            logger.info(f"   帧率: {fps:.2f} fps")
            logger.info(f"   宽高比: {aspect_ratio:.2f}")
            
            # 初始内存状态
            logger.info("🔍 初始内存状态:")
            self._monitor_memory_usage()
            
            # 如果是帧直存格式，直接反归一化并转为帧；否则走神经解码
            if len(frames_or_tokens.shape) == 4:
                # (t, c, h, w), 数值范围[-1,1]
                t, c, h, w = frames_or_tokens.shape
                logger.info(f"🔄 使用帧直还原路径: {t} 帧, {h}x{w}")
                frames = []
                for i in range(t):
                    frame_tensor = torch.from_numpy(frames_or_tokens[i])  # (c,h,w)
                    frame_tensor = torch.clamp(frame_tensor, -1.0, 1.0)
                    frame_tensor = (frame_tensor + 1.0) / 2.0  # [0,1]
                    pil_image = transforms.ToPILImage()(frame_tensor)
                    frames.append(np.array(pil_image))
            else:
                tokens = frames_or_tokens.astype(np.float32)
                frames = self._decode_tokens_to_frames_streaming(tokens, chunk_size=8)
                logger.info(f"✅ 解码生成 {len(frames)} 帧")
            
            # 合成目标分辨率视频
            success = self._frames_to_video_target_resolution(
                frames, output_path, fps, original_width, original_height
            )
            
            if success:
                # 计算目标分辨率
                if self.target_height is None:
                    target_width = original_width
                    target_height = original_height
                else:
                    target_width = int(self.target_height * aspect_ratio)
                    if target_width % 2 != 0:
                        target_width += 1
                    target_height = self.target_height
                
                logger.info(f"✅ 视频解码完成，保存到: {output_path}")
                return {
                    "status": "success",
                    "output_path": output_path,
                    "frames_count": len(frames),
                    "video_info": {
                        "fps": fps,
                        "width": target_width,
                        "height": target_height,
                        "duration": len(frames) / fps,
                        "original_width": original_width,
                        "original_height": original_height,
                        "aspect_ratio": aspect_ratio,
                        "target_height": self.target_height
                    }
                }
            else:
                raise Exception("视频合成失败")
                
        except Exception as e:
            logger.error(f"视频解码失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            raise


async def token2video(
    session_id: str = None,
    token_data: Dict = None,
    output_filename: str = None,
    target_height: int = 480,
    tools_manager = None
) -> dict:
    """
    将token序列转换为高质量视频（480p）
    
    Args:
        session_id: 会话ID
        token_data: token数据字典
        output_filename: 输出视频文件名
        target_height: 目标视频高度（480p）
        tools_manager: 视频工具管理器实例
    
    Returns:
        包含解码结果的字典
    """
    try:
        # 验证参数
        if target_height is not None:
            target_height = max(240, min(target_height, 1080))  # 限制高度范围
        
        resolution_desc = "原视频分辨率" if target_height is None else f"{target_height}p"
        logger.info(f"🎬 开始{resolution_desc}视频解码...")
        logger.info(f"📊 参数: target_height={target_height}")
        
        # 显示系统内存信息
        memory_info = psutil.virtual_memory()
        logger.info(f"💾 系统内存: 总计 {memory_info.total / 1024**3:.1f}GB, 可用 {memory_info.available / 1024**3:.1f}GB")
        
        # 获取token数据
        if session_id:
            session_data = tools_manager.get_session_data(session_id)
            if not session_data:
                return {"status": "error", "message": "会话不存在"}
            
            # 从会话中获取token数据（优先通过管理器按需回填）
            token_data = tools_manager.get_session_data(session_id, "video_tokens")
            # 若内存中没有，尝试从会话目录读取 video_tokens.json（跨请求持久化）
            session_dir = tools_manager.get_session_work_dir(session_id)
            logger.info(f"🔎 会话目录: {session_dir}")
            tokens_json_path = session_dir / "video_tokens.json" if session_dir else None
            if tokens_json_path:
                logger.info(f"🔎 检查tokens文件: {tokens_json_path} 存在={tokens_json_path.exists() if tokens_json_path else False}")
            if not token_data and session_dir and (session_dir / "video_tokens.json").exists():
                try:
                    with open(tokens_json_path, "r", encoding="utf-8") as f:
                        token_data = json.load(f)
                    logger.info("已从磁盘加载 video_tokens.json")
                except Exception as e:
                    logger.warning(f"读取video_tokens.json失败: {e}")
            if not token_data:
                return {"status": "error", "message": "会话中未找到token数据"}
            
            # 设置输出路径
            output_path = session_dir / "exports" / (output_filename or "decoded_video_480p.mp4")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
        elif token_data and output_filename:
            # 直接使用提供的token数据
            temp_dir = Path("temp_outputs")
            temp_dir.mkdir(exist_ok=True)
            output_path = temp_dir / output_filename
        else:
            return {"status": "error", "message": "需要提供 session_id 或 token_data"}
        
        # 验证token数据格式
        required_keys = ["tokens_shape", "tokens_data", "tokens_dtype"]
        if not all(key in token_data for key in required_keys):
            return {"status": "error", "message": "token数据格式不正确"}
        
        # 初始化解码器
        token_dim = token_data["tokens_shape"][1] if len(token_data["tokens_shape"]) > 1 else 256
        logger.info(f"📊 检测到token维度: {token_dim}")
        decoder = VideoDecoder(token_dim=token_dim, target_height=target_height)
        
        # 解码视频
        result = decoder.decode_video(token_data, str(output_path))
        
        # 如果有会话，保存结果信息
        if session_id:
            tools_manager.save_processed_data(session_id, "decoded_video", {
                "output_path": str(output_path),
                "video_info": result["video_info"]
            })
        
        # 计算质量指标
        video_info = result.get("video_info", {})
        quality_info = {
            "resolution": f"{video_info.get('width', 0)}x{video_info.get('height', 0)}",
            "fps": video_info.get('fps', 0),
            "duration": video_info.get('duration', 0),
            "frames_count": result.get('frames_count', 0),
            "aspect_ratio": video_info.get('aspect_ratio', 0),
            "target_height": target_height
        }
        
        return {
            "status": "success",
            "session_id": session_id,
            "output_path": str(output_path),
            "video_info": result["video_info"],
            "quality_info": quality_info,
            "message": f"成功从 {token_data['tokens_shape'][0]} 个token生成{resolution_desc}高质量视频"
        }
    
    except Exception as e:
        logger.error(f"token转视频失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return {"status": "error", "message": f"token转视频失败: {str(e)}"}
