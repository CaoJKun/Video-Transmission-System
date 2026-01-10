import os
import cv2
import uuid
import logging
from pathlib import Path
from tools.video_manager import SUPPORTED_VIDEO_FORMATS

logger = logging.getLogger(__name__)

async def video2pictures(
    session_id: str = None,
    video_file_data: str = None,
    video_filename: str = None,
    max_frames: int = 100,
    tools_manager = None
) -> dict:
    """
    将视频转换为图片帧 - 远程服务器版本
    既保存到服务器又返回给客户端
    
    Args:
        session_id: 会话ID(如果已通过 upload_video 上传)
        video_file_data: base64 编码的视频文件(直接使用)
        video_filename: 视频文件名(直接使用时需要)
        max_frames: 最大提取帧数 (1-1000)
        tools_manager: 视频工具管理器实例
    
    Returns:
        包含图片帧数据的字典，数据既保存在服务器也返回给客户端
    """
    try:
        # 验证参数
        max_frames = max(1, min(max_frames, 1000))  # 限制范围
        
        # 获取视频数据
        temp_video = False
        if session_id:
            session_data = tools_manager.get_session_data(session_id)
            if not session_data:
                return {"status": "error", "message": "会话不存在"}
            
            # 使用会话目录中保存的原始视频文件
            video_path = session_data["original_file"]["path"]
            work_dir = tools_manager.get_session_work_dir(session_id)
            frames_output_dir = work_dir / "frames"
            frames_output_dir.mkdir(exist_ok=True)
            
        elif video_file_data and video_filename:
            extension = Path(video_filename).suffix.lower()
            if extension not in SUPPORTED_VIDEO_FORMATS:
                return {"status": "error", "message": f"不支持的视频格式: {extension}"}
            
            # 保存临时视频文件
            video_path = tools_manager.save_temp_file(video_file_data, extension, session_id)
            frames_output_dir = Path(os.path.dirname(video_path)) / f"frames_{uuid.uuid4().hex[:8]}"
            frames_output_dir.mkdir(exist_ok=True)
            temp_video = True
        else:
            return {"status": "error", "message": "需要提供 session_id 或 video_file_data"}

        if not os.path.exists(video_path):
            return {"status": "error", "message": "视频文件不存在"}
        
        # 提取视频帧
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if temp_video:
                tools_manager.cleanup_temp_files(video_path, str(frames_output_dir))
            return {"status": "error", "message": "无法打开视频文件"}
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames <= 0:
                return {"status": "error", "message": "无法获取视频帧信息"}
            
            # 计算帧间隔
            frame_interval = max(1, total_frames // max_frames)
            
            count = 0
            extracted_count = 0
            frame_files = []
            
            logger.info(f"开始提取视频帧: 总帧数 {total_frames}, 间隔 {frame_interval}, 目标 {max_frames} 帧")
            
            while True:
                ret, frame = cap.read()
                if not ret or extracted_count >= max_frames:
                    break
                
                if count % frame_interval == 0:
                    frame_filename = f"frame_{extracted_count:04d}.jpg"
                    frame_path = frames_output_dir / frame_filename
                    
                    # 优化图片质量和大小 - 适合网络传输
                    success = cv2.imwrite(str(frame_path), frame, [
                        cv2.IMWRITE_JPEG_QUALITY, 85,
                        cv2.IMWRITE_JPEG_OPTIMIZE, 1
                    ])
                    
                    if success and os.path.exists(frame_path):
                        frame_files.append(str(frame_path))
                        extracted_count += 1
                        
                        # 进度日志（每10帧输出一次）
                        if extracted_count % 10 == 0:
                            logger.debug(f"已提取 {extracted_count}/{max_frames} 帧")
                
                count += 1
        finally:
            cap.release()
        
        if not frame_files:
            if temp_video:
                tools_manager.cleanup_temp_files(video_path, str(frames_output_dir))
            return {"status": "error", "message": "未能提取到任何帧"}
        
        logger.info(f"帧提取完成: {extracted_count} 帧")
        
        # 编码图片为 base64 以返回给客户端
        logger.info("正在编码帧数据...")
        encoded_frames = tools_manager.encode_files_to_base64(frame_files)
        
        # 计算帧数据统计
        frames_data = {
            "frame_files": frame_files,
            "output_dir": str(frames_output_dir),
            "encoded_frames": encoded_frames,
            "extraction_info": {
                "total_source_frames": total_frames,
                "frame_interval": frame_interval,
                "fps": round(fps, 2) if fps > 0 else 0,
                "extracted_count": extracted_count
            }
        }
        
        # 如果有会话，保存处理结果到服务器
        if session_id:
            tools_manager.save_processed_data(session_id, "frames", frames_data)
            logger.info(f"帧数据已保存到会话目录: {session_id}")
        
        # 清理临时视频文件（如果是临时的）
        if temp_video:
            tools_manager.cleanup_temp_files(video_path)
        
        # 返回给客户端的完整数据
        return {
            "status": "success",
            "session_id": session_id,
            "total_frames": total_frames,
            "extracted_frames": extracted_count,
            "frame_interval": frame_interval,
            "fps": round(fps, 2) if fps > 0 else 0,
            "frames": encoded_frames,  # 完整的帧数据返回给客户端
            "server_info": {
                "frames_saved_to": str(frames_output_dir),
                "server_frame_count": len(frame_files),
                "work_directory": str(tools_manager.get_session_work_dir(session_id)) if session_id else None
            }
        }
    
    except Exception as e:
        logger.error(f"处理视频帧时出错: {e}")
        # 清理可能的临时文件
        if temp_video and 'video_path' in locals():
            tools_manager.cleanup_temp_files(video_path)
        if 'frames_output_dir' in locals():
            tools_manager.cleanup_temp_files(str(frames_output_dir))
        return {"status": "error", "message": f"处理视频帧时出错: {str(e)}"}