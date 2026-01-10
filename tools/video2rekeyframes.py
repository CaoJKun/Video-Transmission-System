import os
import cv2
import uuid
import logging
from pathlib import Path
from tools.video_manager import SUPPORTED_VIDEO_FORMATS

logger = logging.getLogger(__name__)

async def video2rekeyframes(
    session_id: str = None,
    video_file_data: str = None,
    video_filename: str = None,
    step: int = 5,
    tools_manager = None
) -> dict:
    """
    从视频中提取关键帧到 rekeyframes 文件夹 - 远程服务器版本
    
    Args:
        session_id: 会话ID(如果已通过 upload_video 上传)
        video_file_data: base64 编码的视频文件(直接使用)
        video_filename: 视频文件名(直接使用时需要)
        step: 提取步长，每step帧提取一帧 (默认5，确保大于16帧)
        tools_manager: 视频工具管理器实例
    
    Returns:
        包含关键帧数据的字典
    """
    try:
        # 验证参数
        step = max(1, step)  # 步长至少为1
        
        # 获取视频数据
        temp_video = False
        if session_id:
            session_data = tools_manager.get_session_data(session_id)
            if not session_data:
                return {"status": "error", "message": "会话不存在"}
            
            # 使用会话目录中保存的原始视频文件
            video_path = session_data["original_file"]["path"]
            # 使用固定的 video_processing_data/rekeyframes 目录
            base_dir = Path("video_processing_data")
            rekeyframes_output_dir = base_dir / "rekeyframes"
            rekeyframes_output_dir.mkdir(parents=True, exist_ok=True)
            
        elif video_file_data and video_filename:
            extension = Path(video_filename).suffix.lower()
            if extension not in SUPPORTED_VIDEO_FORMATS:
                return {"status": "error", "message": f"不支持的视频格式: {extension}"}
            
            # 保存临时视频文件
            video_path = tools_manager.save_temp_file(video_file_data, extension, session_id)
            # 使用固定的 video_processing_data/rekeyframes 目录
            base_dir = Path("video_processing_data")
            rekeyframes_output_dir = base_dir / "rekeyframes"
            rekeyframes_output_dir.mkdir(parents=True, exist_ok=True)
            temp_video = True
        else:
            return {"status": "error", "message": "需要提供 session_id 或 video_file_data"}

        if not os.path.exists(video_path):
            return {"status": "error", "message": "视频文件不存在"}
        
        # 提取视频关键帧
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if temp_video:
                tools_manager.cleanup_temp_files(video_path)
            return {"status": "error", "message": "无法打开视频文件"}
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames <= 0:
                return {"status": "error", "message": "无法获取视频帧信息"}
            
            # 清理旧的关键帧文件
            if rekeyframes_output_dir.exists():
                for old_file in rekeyframes_output_dir.glob("rekeyframe_*.jpg"):
                    try:
                        old_file.unlink()
                    except Exception as e:
                        logger.warning(f"清理旧关键帧文件失败: {e}")
            
            count = 0
            extracted_count = 0
            keyframe_files = []
            
            logger.info(f"开始提取视频关键帧: 总帧数 {total_frames}, 步长 {step}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 每step帧提取一帧
                if count % step == 0:
                    keyframe_filename = f"rekeyframe_{extracted_count:04d}.jpg"
                    keyframe_path = rekeyframes_output_dir / keyframe_filename
                    
                    # 优化图片质量和大小
                    success = cv2.imwrite(str(keyframe_path), frame, [
                        cv2.IMWRITE_JPEG_QUALITY, 85,
                        cv2.IMWRITE_JPEG_OPTIMIZE, 1
                    ])
                    
                    if success and os.path.exists(keyframe_path):
                        keyframe_files.append(str(keyframe_path))
                        extracted_count += 1
                        
                        # 进度日志（每10帧输出一次）
                        if extracted_count % 10 == 0:
                            logger.debug(f"已提取 {extracted_count} 个关键帧")
                
                count += 1
        finally:
            cap.release()
        
        if not keyframe_files:
            if temp_video:
                tools_manager.cleanup_temp_files(video_path)
            return {"status": "error", "message": "未能提取到任何关键帧"}
        
        # 检查提取的关键帧数量
        if extracted_count <= 16:
            logger.warning(f"提取的关键帧数量 ({extracted_count}) 小于等于16，建议减小步长")
        
        logger.info(f"关键帧提取完成: {extracted_count} 帧")
        
        # 编码关键帧为 base64 以返回给客户端
        logger.info("正在编码关键帧数据...")
        encoded_keyframes = tools_manager.encode_files_to_base64(keyframe_files)
        
        # 计算关键帧数据统计
        keyframes_data = {
            "keyframe_files": keyframe_files,
            "output_dir": str(rekeyframes_output_dir),
            "encoded_keyframes": encoded_keyframes,
            "extraction_info": {
                "total_source_frames": total_frames,
                "step": step,
                "fps": round(fps, 2) if fps > 0 else 0,
                "extracted_count": extracted_count
            }
        }
        
        # 如果有会话，保存处理结果到服务器
        if session_id:
            tools_manager.save_processed_data(session_id, "rekeyframes", keyframes_data)
            logger.info(f"关键帧数据已保存到会话目录: {session_id}")
        
        # 清理临时视频文件（如果是临时的）
        if temp_video:
            tools_manager.cleanup_temp_files(video_path)
        
        # 返回给客户端的完整数据
        return {
            "status": "success",
            "session_id": session_id,
            "total_frames": total_frames,
            "extracted_keyframes": extracted_count,
            "step": step,
            "fps": round(fps, 2) if fps > 0 else 0,
            "keyframes": encoded_keyframes,  # 完整的关键帧数据返回给客户端
            "server_info": {
                "rekeyframes_saved_to": str(rekeyframes_output_dir),
                "server_keyframe_count": len(keyframe_files),
                "work_directory": str(tools_manager.get_session_work_dir(session_id)) if session_id else None
            }
        }
    
    except Exception as e:
        logger.error(f"处理视频关键帧时出错: {e}")
        # 清理可能的临时文件
        if temp_video and 'video_path' in locals():
            tools_manager.cleanup_temp_files(video_path)
        if 'rekeyframes_output_dir' in locals():
            tools_manager.cleanup_temp_files(str(rekeyframes_output_dir))
        return {"status": "error", "message": f"处理视频关键帧时出错: {str(e)}"}

