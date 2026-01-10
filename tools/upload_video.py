import base64
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

async def upload_video(video_file_data: str, filename: str, tools_manager) -> dict:
    """
    上传视频文件并创建处理会话 - 远程服务器版本
    
    Args:
        video_file_data: base64 编码的视频文件数据
        filename: 视频文件的原始名称
        tools_manager: 视频工具管理器实例
    
    Returns:
        包含 session_id 和工作目录信息的字典，用于后续处理
    """
    try:
        # 验证输入
        if not video_file_data or not filename:
            return {"status": "error", "message": "缺少必要参数: video_file_data 和 filename"}
        
        # 验证文件名安全性（远程服务器安全考虑）
        safe_filename = "".join(c for c in filename if c.isalnum() or c in "._-")
        if safe_filename != filename:
            logger.warning(f"文件名包含特殊字符，已清理: {filename} -> {safe_filename}")
            filename = safe_filename
        
        # 验证 base64 数据
        try:
            decoded_data = base64.b64decode(video_file_data)
            file_size = len(decoded_data)
            
            # 文件大小限制（远程服务器资源考虑）
            max_size = 500 * 1024 * 1024  # 500MB 限制
            if file_size > max_size:
                return {
                    "status": "error", 
                    "message": f"文件过大: {file_size/1024/1024:.1f}MB, 最大允许: {max_size/1024/1024}MB"
                }
                
        except Exception:
            return {"status": "error", "message": "无效的 base64 数据"}
        
        # 创建处理会话
        try:
            session_id = tools_manager.create_session(video_file_data, filename)
            work_directory = tools_manager.get_session_work_dir(session_id)
            
            logger.info(f"视频上传成功: {filename} ({file_size/1024/1024:.2f} MB)")
            logger.info(f"创建会话: {session_id}")
            logger.info(f"工作目录: {work_directory}")
            
            return {
                "status": "success",
                "session_id": session_id,
                "filename": filename,
                "file_size": file_size,
                "file_size_mb": round(file_size / (1024*1024), 2),
                "work_directory": str(work_directory),
                "message": "视频上传成功，会话已创建，可以开始处理",
                "next_steps": [
                    "video2pictures - 提取视频帧",
                    "video2metadata - 获取视频元数据", 
                    "video2audio - 提取音频",
                    "picturepicker - 选择关键帧",
                    "audio2text - 转录音频为文字",
                    "video2motionTokens - 生成运动令牌"
                ]
            }
            
        except ValueError as e:
            return {"status": "error", "message": str(e)}
            
    except Exception as e:
        logger.error(f"上传视频失败: {e}")
        return {"status": "error", "message": f"上传失败: {str(e)}"}