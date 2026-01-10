from datetime import datetime
import os
import sys
import logging
from tools.video_manager import SUPPORTED_VIDEO_FORMATS, SUPPORTED_AUDIO_FORMATS

logger = logging.getLogger(__name__)

def get_session_info(session_id: str, tools_manager) -> dict:
    """获取会话信息和处理状态 - 远程服务器版本"""
    try:
        session_data = tools_manager.get_session_data(session_id)
        if not session_data:
            return {"status": "error", "message": "会话不存在"}
        
        processed_steps = list(session_data["processed_data"].keys())
        original_file = session_data["original_file"]
        created_at = session_data.get("created_at", datetime.now())
        work_dir = session_data.get("work_dir")
        
        # 计算处理进度
        total_steps = ["frames", "keyframes", "audio", "transcript", "metadata", "motion_tokens"]
        progress_percentage = round((len(processed_steps) / len(total_steps)) * 100, 1)
        
        # 计算会话目录大小
        work_dir_size = 0
        file_details = {}
        if work_dir and os.path.exists(work_dir):
            for root, dirs, files in os.walk(work_dir):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    if os.path.exists(filepath):
                        file_size = os.path.getsize(filepath)
                        work_dir_size += file_size
                        
                        # 记录主要文件信息
                        rel_path = os.path.relpath(filepath, work_dir)
                        if any(rel_path.startswith(prefix) for prefix in ['original', 'transcript', 'metadata']):
                            file_details[rel_path] = {
                                "size": file_size,
                                "size_mb": round(file_size / (1024*1024), 3),
                                "modified": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                            }
        
        return {
            "status": "success",
            "session_id": session_id,
            "original_filename": original_file["filename"],
            "file_size": original_file["size"],
            "file_size_mb": round(original_file["size"] / (1024*1024), 2),
            "created_at": created_at.isoformat(),
            "age_minutes": round((datetime.now() - created_at).total_seconds() / 60, 1),
            "work_directory": work_dir,
            "work_dir_size": work_dir_size,
            "work_dir_size_mb": round(work_dir_size / (1024*1024), 2),
            "processed_steps": processed_steps,
            "progress_percentage": progress_percentage,
            "available_data": {
                "frames": "frames" in processed_steps,
                "keyframes": "keyframes" in processed_steps,
                "audio": "audio" in processed_steps,
                "transcript": "transcript" in processed_steps,
                "metadata": "metadata" in processed_steps,
                "motion_tokens": "motion_tokens" in processed_steps
            },
            "file_details": file_details,
            "server_info": {
                "session_type": "remote_processing",
                "auto_cleanup_enabled": True,
                "server_timestamp": datetime.now().isoformat()
            }
        }
    
    except Exception as e:
        logger.error(f"获取会话信息时出错: {e}")
        return {"status": "error", "message": f"获取会话信息时出错: {str(e)}"}

def cleanup_session(session_id: str, tools_manager) -> dict:
    """手动清理会话和相关文件 - 远程服务器版本"""
    try:
        if session_id not in tools_manager.sessions:
            return {"status": "error", "message": "会话不存在"}
        
        # 获取会话信息用于清理报告
        session_data = tools_manager.get_session_data(session_id)
        work_dir = session_data.get("work_dir") if session_data else None
        work_dir_size = 0
        
        if work_dir and os.path.exists(work_dir):
            # 计算要清理的数据大小
            for root, dirs, files in os.walk(work_dir):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    if os.path.exists(filepath):
                        work_dir_size += os.path.getsize(filepath)
        
        # 执行清理
        tools_manager.cleanup_session(session_id)
        
        return {
            "status": "success",
            "message": f"会话 {session_id} 已清理",
            "cleaned_data_size_mb": round(work_dir_size / (1024*1024), 2),
            "cleaned_directory": work_dir,
            "cleanup_timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"清理会话时出错: {e}")
        return {"status": "error", "message": f"清理会话时出错: {str(e)}"}

def get_system_stats(tools_manager, whisper_pipe, validate_ffmpeg, work_dir) -> dict:
    """获取系统统计信息 - 远程服务器版本"""
    try:
        # 会话统计
        session_stats = tools_manager.get_session_stats()
        
        # 磁盘使用统计
        work_dir_size = 0
        session_count_by_status = {"active": 0, "processing": 0, "completed": 0}
        
        try:
            for dirpath, dirnames, filenames in os.walk(work_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        work_dir_size += os.path.getsize(filepath)
            
            # 统计会话状态
            for session_data in tools_manager.sessions.values():
                processed_count = len(session_data.get("processed_data", {}))
                if processed_count == 0:
                    session_count_by_status["active"] += 1
                elif processed_count < 4:
                    session_count_by_status["processing"] += 1
                else:
                    session_count_by_status["completed"] += 1
                    
        except Exception as e:
            logger.warning(f"计算磁盘使用时出错: {e}")
        
        # 系统资源信息
        import psutil
        system_info = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage(str(work_dir)).percent,
            "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 2)
        }
        
        # 系统状态
        system_status = {
            "whisper_loaded": whisper_pipe is not None,
            "ffmpeg_available": validate_ffmpeg(),
            "work_directory": str(work_dir),
            "work_dir_size_mb": round(work_dir_size / (1024*1024), 2),
            "server_uptime": datetime.now().isoformat(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        }
        
        return {
            "status": "success",
            "session_stats": {
                **session_stats,
                "by_status": session_count_by_status
            },
            "system_status": system_status,
            "system_resources": system_info,
            "supported_formats": {
                "video": list(SUPPORTED_VIDEO_FORMATS),
                "audio": list(SUPPORTED_AUDIO_FORMATS)
            },
            "server_info": {
                "deployment_type": "remote_server",
                "max_file_size_mb": 500,
                "session_ttl_hours": 24,
                "auto_cleanup_enabled": True
            }
        }
    
    except Exception as e:
        logger.error(f"获取系统统计时出错: {e}")
        return {"status": "error", "message": f"获取系统统计时出错: {str(e)}"}
