import os
import logging
from datetime import datetime
from pathlib import Path
from tools.utils import validate_ffmpeg, run_ffmpeg_command

logger = logging.getLogger(__name__)

async def video2audio(
    session_id: str = None,
    video_file_data: str = None,
    video_filename: str = None,
    audio_format: str = "wav",
    tools_manager = None
) -> dict:
    """将视频转换为音频 - 远程服务器版本，既保存又返回"""
    try:
        if not validate_ffmpeg():
            return {"status": "error", "message": "ffmpeg 不可用，无法进行音频转换"}
        
        if audio_format not in ["wav", "mp3"]:
            audio_format = "wav"
        
        # 获取视频数据
        temp_video = False
        if session_id:
            session_data = tools_manager.get_session_data(session_id)
            if not session_data:
                return {"status": "error", "message": "会话不存在"}
            
            video_path = session_data["original_file"]["path"]
            work_dir = tools_manager.get_session_work_dir(session_id)
            audio_output_dir = work_dir / "audio"
            audio_output_dir.mkdir(exist_ok=True)
        elif video_file_data and video_filename:
            extension = Path(video_filename).suffix.lower()
            video_path = tools_manager.save_temp_file(video_file_data, extension, session_id)
            audio_output_dir = Path(video_path).parent
            temp_video = True
        else:
            return {"status": "error", "message": "需要提供 session_id 或 video_file_data"}
        
        # 生成音频文件路径
        audio_filename = f"audio.{audio_format}"
        audio_path = audio_output_dir / audio_filename
        
        # 构建 ffmpeg 命令
        if audio_format == "wav":
            cmd = f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_path}"'
        else:  # mp3
            cmd = f'ffmpeg -y -i "{video_path}" -vn -acodec libmp3lame -ar 44100 -ac 2 -b:a 128k "{audio_path}"'
        
        logger.info(f"开始音频提取: {audio_format} 格式")
        returncode, stdout, stderr = run_ffmpeg_command(cmd)
        
        if returncode != 0:
            if temp_video:
                tools_manager.cleanup_temp_files(video_path)
            return {"status": "error", "message": f"音频提取失败: {stderr}"}
        
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            if temp_video:
                tools_manager.cleanup_temp_files(video_path)
            return {"status": "error", "message": "生成的音频文件无效"}
        
        # 编码音频为 base64 返回给客户端
        audio_base64 = tools_manager.encode_file_to_base64(audio_path)
        audio_size = os.path.getsize(audio_path)
        
        logger.info(f"音频提取完成: {audio_size/1024/1024:.2f} MB")
        
        # 准备音频数据
        audio_data = {
            "audio_path": str(audio_path),
            "audio_base64": audio_base64,
            "format": audio_format,
            "size": audio_size,
            "filename": audio_filename
        }
        
        # 如果有会话，保存音频结果到服务器
        if session_id:
            tools_manager.save_processed_data(session_id, "audio", audio_data)
            logger.info(f"音频已保存到服务器: {audio_path}")
        
        # 清理临时视频文件
        if temp_video:
            tools_manager.cleanup_temp_files(video_path)
        
        # 返回给客户端的完整数据
        return {
            "status": "success",
            "session_id": session_id,
            "audio_data": audio_base64,  # 完整音频数据返回给客户端
            "audio_filename": audio_filename,
            "format": audio_format,
            "size": audio_size,
            "size_mb": round(audio_size / (1024*1024), 3),
            "server_info": {
                "audio_saved_to": str(audio_path),
                "extraction_completed_at": datetime.now().isoformat()
            }
        }
    
    except Exception as e:
        logger.error(f"视频转音频时出错: {e}")
        return {"status": "error", "message": f"视频转音频时出错: {str(e)}"}
