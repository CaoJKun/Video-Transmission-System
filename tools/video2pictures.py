import os
import subprocess
import json
import re
import shutil
import math
import logging
from pathlib import Path
from tools.video_manager import SUPPORTED_VIDEO_FORMATS

logger = logging.getLogger(__name__)

def get_video_info(input_video):
    """获取视频的详细信息"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,codec_name,bit_rate',
            '-of', 'json',
            input_video
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return None
        
        video_data = json.loads(result.stdout)
        if 'streams' not in video_data or len(video_data['streams']) == 0:
            return None
        
        stream = video_data['streams'][0]
        
        # 解析帧率
        frame_rate_str = stream.get('r_frame_rate', '25/1')
        if '/' in frame_rate_str:
            num, den = map(int, frame_rate_str.split('/'))
            frame_rate = num / den if den != 0 else 25.0
        else:
            frame_rate = float(frame_rate_str)
        
        # 获取视频时长
        cmd_duration = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            input_video
        ]
        result_duration = subprocess.run(cmd_duration, capture_output=True, text=True)
        duration = float(result_duration.stdout.strip()) if result_duration.returncode == 0 else None
        
        info = {
            'width': int(stream.get('width', 1280)),
            'height': int(stream.get('height', 720)),
            'frame_rate': frame_rate,
            'codec': stream.get('codec_name', 'hevc'),
            'bit_rate': stream.get('bit_rate', None),
            'duration': duration
        }
        
        # 获取音频信息
        cmd_audio = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_name,sample_rate,channels',
            '-of', 'json',
            input_video
        ]
        result_audio = subprocess.run(cmd_audio, capture_output=True, text=True)
        
        if result_audio.returncode == 0:
            audio_data = json.loads(result_audio.stdout)
            if 'streams' in audio_data and len(audio_data['streams']) > 0:
                audio_stream = audio_data['streams'][0]
                info['has_audio'] = True
                info['audio_codec'] = audio_stream.get('codec_name', 'aac')
            else:
                info['has_audio'] = False
        else:
            info['has_audio'] = False
        
        return info
    except Exception as e:
        logger.warning(f"无法获取视频信息，使用默认值: {str(e)}")
        return {
            'width': 1280,
            'height': 672,
            'frame_rate': 25.0,
            'codec': 'hevc',
            'has_audio': True,
            'audio_codec': 'aac',
            'duration': None
        }

def calculate_frame_difference(frame1_path, frame2_path):
    """计算两帧之间的差异（使用ffmpeg的ssim）"""
    try:
        cmd = [
            'ffmpeg',
            '-i', frame1_path,
            '-i', frame2_path,
            '-lavfi', 'ssim',
            '-f', 'null',
            '-'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            for line in result.stderr.split('\n'):
                if 'All:' in line:
                    try:
                        ssim_match = re.search(r'All:([\d.]+)', line)
                        if ssim_match:
                            ssim_value = float(ssim_match.group(1))
                            return 1.0 - ssim_value
                    except:
                        pass
        
        return 0.5
    except Exception as e:
        return 0.5

def extract_keyframes_adaptive(input_video, output_frames_dir, num_keyframes=16):
    """根据画面变化程度自适应提取关键帧"""
    try:
        if not os.path.exists(output_frames_dir):
            os.makedirs(output_frames_dir)
        
        # 清空文件夹
        for file in os.listdir(output_frames_dir):
            if file.endswith(('.jpg', '.png', '.jpeg')):
                os.remove(os.path.join(output_frames_dir, file))
        
        # 使用ffmpeg的scene detection来检测场景变化
        temp_dir = os.path.join(output_frames_dir, "temp_analysis")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # 获取视频时长
        cmd_duration = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            input_video
        ]
        result_duration = subprocess.run(cmd_duration, capture_output=True, text=True)
        duration = float(result_duration.stdout.strip()) if result_duration.returncode == 0 else None
        
        # 提取足够多的候选帧
        candidate_count = max(num_keyframes * 8, 150)
        candidate_fps = candidate_count / duration if duration else 8.0
        
        temp_pattern = os.path.join(temp_dir, "frame_%06d.jpg")
        
        logger.info(f"正在提取候选帧用于分析（采样率: {candidate_fps:.2f} fps）...")
        cmd_extract = [
            'ffmpeg',
            '-i', input_video,
            '-vf', f"fps={candidate_fps}",
            '-qscale:v', '2',
            temp_pattern,
            '-y'
        ]
        
        result = subprocess.run(cmd_extract, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg 错误: {result.stderr}")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return False, 0
        
        # 获取所有提取的帧
        all_frames = sorted([f for f in os.listdir(temp_dir) 
                            if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        if len(all_frames) < 2:
            logger.error("提取的帧数不足")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return False, 0
        
        logger.info(f"提取了 {len(all_frames)} 帧候选帧")
        logger.info("正在计算帧间差异...")
        
        # 计算相邻帧之间的差异
        frame_differences = []
        for i in range(len(all_frames) - 1):
            frame1_path = os.path.join(temp_dir, all_frames[i])
            frame2_path = os.path.join(temp_dir, all_frames[i + 1])
            diff = calculate_frame_difference(frame1_path, frame2_path)
            frame_differences.append({
                'index': i,
                'difference': diff,
                'frame': all_frames[i]
            })
            if (i + 1) % 20 == 0:
                logger.debug(f"已分析 {i + 1}/{len(all_frames) - 1} 对帧")
        
        # 计算差异的统计信息
        differences = [d['difference'] for d in frame_differences]
        if not differences:
            logger.error("无法计算帧差异")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return False, 0
        
        avg_diff = sum(differences) / len(differences)
        max_diff = max(differences)
        min_diff = min(differences)
        std_diff = math.sqrt(sum((d - avg_diff) ** 2 for d in differences) / len(differences)) if len(differences) > 1 else 0
        
        logger.info(f"差异统计: 平均={avg_diff:.4f}, 最大={max_diff:.4f}, 最小={min_diff:.4f}, 标准差={std_diff:.4f}")
        
        # 根据差异大小分配关键帧
        selected_frames = []
        selected_indices = set()
        
        # 首先选择第一帧和最后一帧
        selected_frames.append((0, all_frames[0]))
        selected_indices.add(0)
        if len(all_frames) > 1:
            selected_frames.append((len(all_frames) - 1, all_frames[-1]))
            selected_indices.add(len(all_frames) - 1)
        
        # 将视频分成多个区间，根据每个区间的平均差异来分配帧数
        remaining_slots = num_keyframes - len(selected_frames)
        
        if remaining_slots > 0:
            interval_size = len(frame_differences) // remaining_slots
            interval_differences = []
            
            for i in range(remaining_slots):
                start_idx = i * interval_size
                end_idx = min((i + 1) * interval_size, len(frame_differences))
                interval_diff = sum(frame_differences[j]['difference'] for j in range(start_idx, end_idx))
                interval_differences.append({
                    'start': start_idx,
                    'end': end_idx,
                    'diff_sum': interval_diff,
                    'avg_diff': interval_diff / (end_idx - start_idx) if end_idx > start_idx else 0
                })
            
            total_diff = sum(interval['diff_sum'] for interval in interval_differences)
            
            allocated_frames = []
            for interval in interval_differences:
                if total_diff > 0:
                    weight = interval['diff_sum'] / total_diff
                    num_frames_for_interval = max(1, int(weight * remaining_slots))
                else:
                    num_frames_for_interval = 1
                
                interval_frames = [fd for fd in frame_differences[interval['start']:interval['end']]]
                interval_frames.sort(key=lambda x: x['difference'], reverse=True)
                
                for diff_info in interval_frames[:num_frames_for_interval]:
                    if diff_info['index'] not in selected_indices and len(allocated_frames) < remaining_slots:
                        allocated_frames.append((diff_info['index'], all_frames[diff_info['index']]))
                        selected_indices.add(diff_info['index'])
            
            selected_frames.extend(allocated_frames)
        
        # 如果选择的帧数还不够，从差异最大的帧中补充
        if len(selected_frames) < num_keyframes:
            sorted_diffs = sorted(frame_differences, key=lambda x: x['difference'], reverse=True)
            for diff_info in sorted_diffs:
                if len(selected_frames) >= num_keyframes:
                    break
                if diff_info['index'] not in selected_indices:
                    selected_frames.append((diff_info['index'], all_frames[diff_info['index']]))
                    selected_indices.add(diff_info['index'])
        
        # 如果还是不够，均匀补充
        if len(selected_frames) < num_keyframes:
            step = len(all_frames) / (num_keyframes - len(selected_frames) + 1)
            for i in range(1, num_keyframes - len(selected_frames) + 1):
                idx = int(i * step)
                if idx not in selected_indices and idx < len(all_frames):
                    selected_frames.append((idx, all_frames[idx]))
                    selected_indices.add(idx)
        
        # 按索引排序选中的帧
        selected_frames.sort(key=lambda x: x[0])
        selected_frame_files = [f[1] for f in selected_frames]
        
        # 复制选中的帧到输出目录
        output_pattern = os.path.join(output_frames_dir, "keyframe_%04d.jpg")
        for idx, frame_file in enumerate(selected_frame_files, 1):
            src = os.path.join(temp_dir, frame_file)
            dst = output_pattern % idx
            shutil.copy2(src, dst)
        
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        frame_count = len(selected_frame_files)
        logger.info(f"自适应关键帧提取完成，共提取 {frame_count} 个关键帧")
        return True, frame_count
        
    except Exception as e:
        logger.error(f"自适应提取关键帧时出错: {str(e)}")
        temp_dir = os.path.join(output_frames_dir, "temp_analysis")
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        return False, 0

async def video2pictures(
    session_id: str = None,
    video_file_data: str = None,
    video_filename: str = None,
    num_keyframes: int = 48,
    tools_manager = None
) -> dict:
    """
    从视频中自适应提取关键帧到 keyframes 文件夹 - 远程服务器版本
    
    Args:
        session_id: 会话ID(如果已通过 upload_video 上传)
        video_file_data: base64 编码的视频文件(直接使用)
        video_filename: 视频文件名(直接使用时需要)
        num_keyframes: 要提取的关键帧数量（默认48）
        tools_manager: 视频工具管理器实例
    
    Returns:
        包含关键帧数据的字典
    """
    try:
        # 验证参数
        num_keyframes = max(16, min(num_keyframes, 200))  # 至少16帧，最多200帧
        
        # 获取视频数据
        temp_video = False
        if session_id:
            session_data = tools_manager.get_session_data(session_id)
            if not session_data:
                return {"status": "error", "message": "会话不存在"}
            
            video_path = session_data["original_file"]["path"]
            work_dir = tools_manager.get_session_work_dir(session_id)
            keyframes_output_dir = work_dir / "keyframes"
            keyframes_output_dir.mkdir(exist_ok=True)
            
        elif video_file_data and video_filename:
            extension = Path(video_filename).suffix.lower()
            if extension not in SUPPORTED_VIDEO_FORMATS:
                return {"status": "error", "message": f"不支持的视频格式: {extension}"}
            
            video_path = tools_manager.save_temp_file(video_file_data, extension, session_id)
            base_dir = Path("video_processing_data")
            keyframes_output_dir = base_dir / "keyframes"
            keyframes_output_dir.mkdir(parents=True, exist_ok=True)
            temp_video = True
        else:
            return {"status": "error", "message": "需要提供 session_id 或 video_file_data"}

        if not os.path.exists(video_path):
            return {"status": "error", "message": "视频文件不存在"}
        
        # 获取视频信息
        video_info = get_video_info(video_path)
        
        # 使用自适应方法提取关键帧
        logger.info(f"开始自适应提取关键帧: 目标 {num_keyframes} 帧")
        success, frame_count = extract_keyframes_adaptive(str(video_path), str(keyframes_output_dir), num_keyframes)
        
        if not success or frame_count == 0:
            if temp_video:
                tools_manager.cleanup_temp_files(video_path)
            return {"status": "error", "message": "关键帧提取失败"}
        
        # 获取关键帧文件列表
        keyframe_files = sorted([str(keyframes_output_dir / f) for f in os.listdir(keyframes_output_dir) 
                                 if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        # 编码关键帧为 base64
        encoded_keyframes = tools_manager.encode_files_to_base64(keyframe_files)
        
        # 准备返回数据
        result_data = {
            "status": "success",
            "session_id": session_id,
            "total_frames": video_info.get('frame_count', 0) if video_info else 0,
            "extracted_keyframes": frame_count,
            "num_keyframes": num_keyframes,
            "keyframes": encoded_keyframes,
            "video_info": video_info,
            "server_info": {
                "keyframes_saved_to": str(keyframes_output_dir),
                "server_keyframe_count": len(keyframe_files)
            }
        }
        
        # 如果有会话，保存处理结果
        if session_id:
            tools_manager.save_processed_data(session_id, "keyframes", result_data)
            logger.info(f"关键帧数据已保存到会话目录: {session_id}")
        
        # 清理临时视频文件
        if temp_video:
            tools_manager.cleanup_temp_files(video_path)
        
        return result_data
        
    except Exception as e:
        logger.error(f"处理视频关键帧时出错: {e}")
        if temp_video and 'video_path' in locals():
            tools_manager.cleanup_temp_files(video_path)
        return {"status": "error", "message": f"处理视频关键帧时出错: {str(e)}"}
