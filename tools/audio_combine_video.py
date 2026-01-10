import os
import logging
from pathlib import Path
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, concatenate_audioclips

logger = logging.getLogger(__name__)

async def audio_combine_video(
    session_id: str = None,
    audio_file_path: str = None,
    video_file_path: str = None,
    output_filename: str = "combined_video.mp4",
    tools_manager = None
) -> dict:
    """
    使用 MoviePy 将音频和视频结合 - 远程服务器版本
    
    Args:
        session_id: 会话ID(如果已通过其他工具处理)
        audio_file_path: 音频文件路径(如果直接指定)
        video_file_path: 视频文件路径(如果直接指定)
        output_filename: 输出文件名
        tools_manager: 视频工具管理器实例
    
    Returns:
        包含处理结果的字典
    """
    try:
        # 验证参数
        if not output_filename.endswith('.mp4'):
            output_filename += '.mp4'
        
        # 获取文件路径
        if session_id:
            session_data = tools_manager.get_session_data(session_id)
            if not session_data:
                return {"status": "error", "message": "会话不存在"}
            
            work_dir = tools_manager.get_session_work_dir(session_id)
            
            # 默认使用会话目录中的文件
            audio_path = work_dir / "audio" / "audio.wav"
            video_path = work_dir / "exports" / "decoded_test_video_480p.mp4"
            output_path = work_dir / "exports" / output_filename
            
        elif audio_file_path and video_file_path:
            audio_path = Path(audio_file_path)
            video_path = Path(video_file_path)
            output_path = Path(audio_file_path).parent / output_filename
        else:
            return {"status": "error", "message": "需要提供 session_id 或 audio_file_path 和 video_file_path"}
        
        # 检查文件是否存在
        if not audio_path.exists():
            return {"status": "error", "message": f"音频文件不存在: {audio_path}"}
        
        if not video_path.exists():
            return {"status": "error", "message": f"视频文件不存在: {video_path}"}
        
        logger.info(f"开始结合音频和视频: 音频={audio_path}, 视频={video_path}")
        
        # 使用 MoviePy 处理
        try:
            # 加载视频和音频
            video_clip = VideoFileClip(str(video_path))
            audio_clip = AudioFileClip(str(audio_path))
            
            logger.info(f"视频时长: {video_clip.duration:.2f}秒, 音频时长: {audio_clip.duration:.2f}秒")
            
            # 调整音频长度以匹配视频
            if audio_clip.duration > video_clip.duration:
                # 如果音频比视频长，截取音频
                audio_clip = audio_clip.subclip(0, video_clip.duration)
                logger.info("音频已截取以匹配视频长度")
            elif audio_clip.duration < video_clip.duration:
                # 如果音频比视频短，使用循环填充
                loops_needed = int(video_clip.duration / audio_clip.duration) + 1
                # 使用 concatenate_audioclips 来循环音频
                audio_clips = [audio_clip] * loops_needed
                audio_clip = concatenate_audioclips(audio_clips).subclip(0, video_clip.duration)
                logger.info("音频已循环以匹配视频长度")
            
            # 将音频添加到视频
            final_clip = video_clip.set_audio(audio_clip)
            
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 导出最终视频
            logger.info(f"正在导出到: {output_path}")
            final_clip.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None
            )
            
            # 清理资源
            video_clip.close()
            audio_clip.close()
            final_clip.close()
            
            logger.info(f"音频视频结合完成: {output_path}")
            
            # 获取输出文件信息
            output_size = output_path.stat().st_size if output_path.exists() else 0
            
            # 如果有会话，保存处理结果
            if session_id:
                combine_data = {
                    "audio_file": str(audio_path),
                    "video_file": str(video_path),
                    "output_file": str(output_path),
                    "output_size": output_size,
                    "processing_info": {
                        "original_video_duration": video_clip.duration,
                        "original_audio_duration": audio_clip.duration,
                        "final_duration": final_clip.duration
                    }
                }
                tools_manager.save_processed_data(session_id, "audio_combined", combine_data)
                logger.info(f"处理结果已保存到会话: {session_id}")
            
            return {
                "status": "success",
                "session_id": session_id,
                "audio_file": str(audio_path),
                "video_file": str(video_path),
                "output_file": str(output_path),
                "output_size": output_size,
                "processing_info": {
                    "original_video_duration": video_clip.duration,
                    "original_audio_duration": audio_clip.duration,
                    "final_duration": final_clip.duration
                },
                "server_info": {
                    "output_saved_to": str(output_path),
                    "processing_completed_at": logger.info("处理完成")
                }
            }
            
        except Exception as e:
            logger.error(f"MoviePy 处理过程中出错: {e}")
            return {"status": "error", "message": f"视频处理失败: {str(e)}"}
    
    except Exception as e:
        logger.error(f"音频视频结合时出错: {e}")
        return {"status": "error", "message": f"音频视频结合时出错: {str(e)}"}

