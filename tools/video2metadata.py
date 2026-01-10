import base64
import os
import cv2
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

async def video2metadata(
    session_id: str = None,
    video_file_data: str = None,
    video_filename: str = None,
    tools_manager = None
) -> dict:
    """获取视频元数据 - 远程服务器版本，既保存又返回"""
    try:
        # 获取视频数据
        temp_video = False
        if session_id:
            session_data = tools_manager.get_session_data(session_id)
            if not session_data:
                return {"status": "error", "message": "会话不存在"}
            
            video_path = session_data["original_file"]["path"]
            original_filename = session_data["original_file"]["filename"]
            work_dir = tools_manager.get_session_work_dir(session_id)
        elif video_file_data and video_filename:
            extension = Path(video_filename).suffix.lower()
            video_path = tools_manager.save_temp_file(video_file_data, extension, session_id)
            original_filename = video_filename
            temp_video = True
            work_dir = None
        else:
            return {"status": "error", "message": "需要提供 session_id 或 video_file_data"}
        
        # 提取元数据
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if temp_video:
                tools_manager.cleanup_temp_files(video_path)
            return {"status": "error", "message": "无法打开视频文件"}
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 计算时长和文件大小
            duration = frame_count / fps if fps > 0 else 0
            if session_id:
                file_size = os.path.getsize(video_path)
            else:
                file_size = len(base64.b64decode(video_file_data))
            
            # 计算比特率
            bitrate = (file_size * 8) / duration if duration > 0 else 0
            
            metadata = {
                "filename": original_filename,
                "format": Path(original_filename).suffix.lstrip('.').upper(),
                "fps": round(fps, 2),
                "width": width,
                "height": height,
                "frame_count": frame_count,
                "duration": round(duration, 2),
                "duration_formatted": f"{int(duration//3600):02d}:{int((duration%3600)//60):02d}:{int(duration%60):02d}",
                "resolution": f"{width}x{height}",
                "aspect_ratio": round(width/height, 2) if height > 0 else 0,
                "file_size": file_size,
                "file_size_mb": round(file_size / (1024*1024), 2),
                "bitrate_kbps": round(bitrate / 1000, 2) if bitrate > 0 else 0,
                "extraction_timestamp": datetime.now().isoformat()
            }
        finally:
            cap.release()
        
        logger.info(f"元数据提取完成: {metadata['resolution']}, {metadata['duration_formatted']}")
        
        # 如果有会话，保存元数据到服务器
        if session_id:
            tools_manager.save_processed_data(session_id, "metadata", metadata)
            
            # 额外保存为JSON文件
            if work_dir:
                metadata_file = work_dir / "metadata.json"
                try:
                    with open(metadata_file, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
                    logger.info(f"元数据已保存到服务器: {metadata_file}")
                except Exception as e:
                    logger.error(f"保存元数据文件失败: {e}")
        
        # 清理临时文件
        if temp_video:
            tools_manager.cleanup_temp_files(video_path)
        
        # 返回给客户端的完整元数据
        return {
            "status": "success",
            "session_id": session_id,
            **metadata,  # 所有元数据返回给客户端
            "server_info": {
                "metadata_saved_to": str(work_dir / "metadata.json") if work_dir else None,
                "extraction_completed_at": datetime.now().isoformat()
            }
        }
    
    except Exception as e:
        logger.error(f"获取视频元数据时出错: {e}")
        return {"status": "error", "message": f"获取视频元数据时出错: {str(e)}"}