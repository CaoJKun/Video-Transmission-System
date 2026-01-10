import os
import subprocess
import json
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

def create_video_from_keyframes(frames_dir, output_video, original_video=None, video_info=None):
    """从关键帧文件夹生成新视频，在关键帧之间进行插帧，保持与原视频相同的参数和时长"""
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_video)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 检查关键帧文件
        frame_files = sorted([f for f in os.listdir(frames_dir) 
                      if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        if not frame_files:
            logger.error("关键帧文件夹为空，无法生成视频")
            return False
        
        keyframe_count = len(frame_files)
        logger.info(f"关键帧数量: {keyframe_count}")
        
        # 如果没有提供视频信息，尝试从原视频获取
        if video_info is None and original_video and os.path.exists(original_video):
            logger.info("正在获取原视频参数...")
            video_info = get_video_info(original_video)
        
        # 如果仍然无法获取，使用默认值
        if video_info is None:
            video_info = {
                'width': 1280,
                'height': 672,
                'frame_rate': 25.0,
                'codec': 'hevc',
                'has_audio': True,
                'audio_codec': 'aac',
                'duration': None
            }
            logger.info("使用默认视频参数")
        else:
            logger.info(f"视频参数: {video_info['width']}x{video_info['height']}, "
                  f"{video_info['frame_rate']:.2f}fps, 编码: {video_info['codec']}")
            if video_info.get('duration'):
                logger.info(f"原视频时长: {video_info['duration']:.2f} 秒")
        
        # 计算需要的总帧数
        if video_info.get('duration') and video_info.get('frame_rate'):
            total_frames_needed = int(video_info['duration'] * video_info['frame_rate'])
            logger.info(f"需要总帧数: {total_frames_needed} (基于 {video_info['duration']:.2f}秒 x {video_info['frame_rate']:.2f}fps)")
        else:
            total_frames_needed = keyframe_count * int(video_info['frame_rate'])
            logger.warning(f"无法获取原视频时长，使用估算值: {total_frames_needed} 帧")
        
        # 构建输入路径模式
        input_pattern = os.path.join(frames_dir, "keyframe_%04d.jpg")
        
        # 确定视频编码器
        if video_info['codec'].lower() in ['hevc', 'h265', 'libx265']:
            video_codec = 'libx265'
        else:
            video_codec = 'libx264'
        
        # 计算输入帧率：让关键帧均匀分布在原视频时长内
        if video_info.get('duration'):
            input_framerate = keyframe_count / video_info['duration']
            logger.info(f"输入帧率: {input_framerate:.4f} fps (关键帧均匀分布)")
        else:
            input_framerate = 1.0
            logger.warning(f"无法获取原视频时长，使用默认输入帧率: {input_framerate} fps")
        
        # 构建ffmpeg命令
        cmd = [
            'ffmpeg',
            '-framerate', str(input_framerate),
            '-i', input_pattern
        ]
        
        # 如果有原视频且包含音频，添加音频输入
        audio_file = None
        if original_video and os.path.exists(original_video) and video_info.get('has_audio', False):
            try:
                audio_file = os.path.join(output_dir, "temp_audio.aac")
                audio_cmd = [
                    'ffmpeg',
                    '-i', original_video,
                    '-vn',
                    '-acodec', 'copy',
                    audio_file,
                    '-y'
                ]
                audio_result = subprocess.run(audio_cmd, capture_output=True, text=True)
                
                if audio_result.returncode == 0 and os.path.exists(audio_file):
                    cmd.extend(['-i', audio_file])
                    logger.info("将合并原视频音频")
                else:
                    logger.warning("无法提取音频，将生成无音频视频")
                    audio_file = None
            except Exception as e:
                logger.warning(f"音频处理失败，将生成无音频视频: {str(e)}")
                audio_file = None
        
        # 构建视频滤镜链：先缩放，然后进行运动插帧
        filter_chain = [
            f"scale={video_info['width']}:{video_info['height']}",
            f"minterpolate=fps={video_info['frame_rate']}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:me=umh:vsbmc=1:scd=none:mb_size=8:search_param=48",
            f"atadenoise=0a=0.01:0b=0.02:1a=0.01:1b=0.02:2a=0.01:2b=0.02:s=7"
        ]
        
        vf_string = ','.join(filter_chain)
        
        # 添加输出选项
        cmd.extend([
            '-vf', vf_string,
            '-c:v', video_codec,
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            '-preset', 'medium',
            '-r', str(video_info['frame_rate'])
        ])
        
        # 如果有音频，添加音频编码选项
        if audio_file and os.path.exists(audio_file):
            cmd.extend(['-c:a', video_info.get('audio_codec', 'aac')])
            if video_info.get('duration'):
                cmd.extend(['-t', str(video_info['duration'])])
            else:
                cmd.extend(['-shortest'])
        else:
            if video_info.get('duration'):
                cmd.extend(['-t', str(video_info['duration'])])
        
        # 添加输出文件
        cmd.extend([output_video, '-y'])
        
        # 执行ffmpeg命令
        logger.info("正在生成视频（关键帧插帧中）...")
        logger.info(f"使用高质量运动插帧算法（8像素块+48像素搜索+不均匀多六边形搜索+时间平滑）")
        logger.info(f"关键帧数量: {keyframe_count}，将生成 {total_frames_needed} 帧，确保流畅连续")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 清理临时音频文件
        if audio_file and os.path.exists(audio_file):
            os.remove(audio_file)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg 错误: {result.stderr}")
            return False
        
        # 验证生成的视频时长
        if video_info.get('duration'):
            cmd_check = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                output_video
            ]
            result_check = subprocess.run(cmd_check, capture_output=True, text=True)
            if result_check.returncode == 0:
                generated_duration = float(result_check.stdout.strip())
                logger.info(f"生成的视频时长: {generated_duration:.2f} 秒")
                if abs(generated_duration - video_info['duration']) < 0.5:
                    logger.info(f"视频时长匹配原视频 (差异: {abs(generated_duration - video_info['duration']):.2f}秒)")
                else:
                    logger.warning(f"视频时长略有差异 (差异: {abs(generated_duration - video_info['duration']):.2f}秒)")
        
        logger.info(f"视频生成完成: {output_video}")
        return True
        
    except FileNotFoundError:
        logger.error("找不到 ffmpeg 命令，请确保已安装 ffmpeg 并添加到 PATH")
        return False
    except Exception as e:
        logger.error(f"生成视频时出错: {str(e)}")
        return False

async def pictures2video(
    session_id: str = None,
    frames_data: list = None,
    output_filename: str = None,
    fps: float = 30.0,
    codec: str = "mp4v",
    quality: int = 90,
    tools_manager = None
) -> dict:
    """
    使用OpenCV将图片帧转换为视频（兼容性函数）
    
    Args:
        session_id: 会话ID，用于获取帧数据
        frames_data: 帧数据列表（base64编码的图片）
        output_filename: 输出视频文件名
        fps: 视频帧率
        codec: 视频编码器
        quality: 视频质量 (0-100)
        tools_manager: 视频工具管理器实例
    
    Returns:
        包含生成视频信息的字典
    """
    # 如果有session_id，使用pictures2video_from_session
    if session_id and tools_manager:
        return await pictures2video_from_session(session_id, output_filename, tools_manager)
    else:
        # 如果没有session_id，返回错误（需要session_id才能使用新的插帧功能）
        return {"status": "error", "message": "需要提供session_id才能使用此功能"}

async def pictures2video_from_session(
    session_id: str,
    output_filename: str = None,
    tools_manager = None
) -> dict:
    """
    从会话的关键帧数据生成视频，使用运动插帧保持原视频参数和时长
    
    Args:
        session_id: 会话ID
        output_filename: 输出视频文件名
        tools_manager: 视频工具管理器实例
    
    Returns:
        包含生成视频信息的字典
    """
    try:
        if not session_id or not tools_manager:
            return {"status": "error", "message": "需要提供有效的会话ID和工具管理器"}
        
        work_dir = tools_manager.get_session_work_dir(session_id)
        keyframes_dir = work_dir / "keyframes"
        
        logger.info(f"工作目录: {work_dir}")
        logger.info(f"关键帧目录: {keyframes_dir}")
        
        if not keyframes_dir.exists():
            return {"status": "error", "message": f"关键帧文件夹不存在: {keyframes_dir}"}
        
        # 检查关键帧文件
        frame_files = sorted([f for f in keyframes_dir.iterdir() 
                             if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        if not frame_files:
            return {"status": "error", "message": f"在 {keyframes_dir} 中未找到关键帧文件"}
        
        logger.info(f"找到 {len(frame_files)} 个关键帧")
        
        # 获取原视频信息
        session_data = tools_manager.get_session_data(session_id)
        original_video_path = session_data.get("original_file", {}).get("path") if session_data else None
        
        metadata_info = tools_manager.get_session_data(session_id, "metadata")
        if not metadata_info:
            return {"status": "error", "message": "无法获取原视频元数据"}
        
        # 构建视频信息
        video_info = {
            'width': metadata_info.get("width", 1280),
            'height': metadata_info.get("height", 672),
            'frame_rate': metadata_info.get("fps", 25.0),
            'codec': 'libx264',
            'has_audio': True,
            'audio_codec': 'aac',
            'duration': metadata_info.get("duration", None)
        }
        
        # 设置输出路径
        if not output_filename:
            output_filename = "frames_to_video_output.mp4"
        output_path = work_dir / "exports" / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"开始从关键帧生成视频")
        logger.info(f"输出路径: {output_path}")
        
        # 从关键帧生成视频
        success = create_video_from_keyframes(
            str(keyframes_dir),
            str(output_path),
            original_video=original_video_path,
            video_info=video_info
        )
        
        if not success:
            return {"status": "error", "message": "视频生成失败"}
        
        # 检查输出文件是否生成成功
        if not os.path.exists(output_path):
            return {"status": "error", "message": "视频文件生成失败"}
        
        # 获取视频文件信息
        video_size = output_path.stat().st_size if output_path.exists() else 0
        
        # 获取视频元数据
        from cv2 import VideoCapture, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT
        cap = VideoCapture(str(output_path))
        if cap.isOpened():
            actual_fps = cap.get(CAP_PROP_FPS)
            frame_count = int(cap.get(CAP_PROP_FRAME_COUNT))
            duration = frame_count / actual_fps if actual_fps > 0 else 0
            cap.release()
        else:
            actual_fps = video_info['frame_rate']
            frame_count = int(video_info['duration'] * video_info['frame_rate']) if video_info.get('duration') else 0
            duration = video_info.get('duration', 0)
        
        logger.info(f"视频生成完成: {output_path} ({video_size/1024/1024:.2f} MB)")
        logger.info(f"视频信息: {frame_count}帧, {actual_fps:.2f}fps, {duration:.2f}秒")
        
        # 准备返回数据
        result_data = {
            "status": "success",
            "session_id": session_id,
            "output_path": str(output_path),
            "output_filename": output_filename,
            "video_info": {
                "width": video_info['width'],
                "height": video_info['height'],
                "channels": 3,
                "fps": actual_fps,
                "frame_count": frame_count,
                "duration": round(duration, 2),
                "size": video_size,
                "size_mb": round(video_size / (1024*1024), 3)
            },
            "processing_info": {
                "input_keyframes": len(frame_files),
                "processed_frames": frame_count,
                "codec": video_info.get('codec', 'libx264'),
                "target_fps": video_info['frame_rate'],
                "original_duration": video_info.get('duration'),
                "interpolation": True
            }
        }
        
        # 保存结果到会话数据
        tools_manager.save_processed_data(session_id, "frames_to_video", result_data)
        logger.info(f"视频生成结果已保存到会话: {session_id}")
        
        return result_data
    
    except Exception as e:
        logger.error(f"从关键帧生成视频时出错: {e}")
        return {"status": "error", "message": f"从关键帧生成视频时出错: {str(e)}"}
