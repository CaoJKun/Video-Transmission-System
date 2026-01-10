import os
import logging
from datetime import datetime
from pathlib import Path
from tools.video_manager import SUPPORTED_AUDIO_FORMATS

logger = logging.getLogger(__name__)

async def audio2text(
    session_id: str = None,
    audio_file_data: str = None,
    audio_filename: str = None,
    language: str = "auto",
    tools_manager = None,
    whisper_pipe = None
) -> dict:
    """
    将音频转换为文字 - 远程服务器版本
    既保存到服务器又返回给客户端
    
    Args:
        session_id: 会话ID(使用之前 video2audio 的结果)
        audio_file_data: base64 编码的音频文件
        audio_filename: 音频文件名
        language: 语言代码 ("auto", "en", "zh", "ja", etc.)
        tools_manager: 视频工具管理器实例
        whisper_pipe: Whisper 模型实例
    
    Returns:
        包含转录文字的字典，同时保存到服务器
    """
    try:
        if not whisper_pipe:
            return {
                "status": "error", 
                "message": "Whisper 模型未加载，请稍后重试或联系管理员",
                "retry_suggested": True
            }
        
        # 获取音频数据
        temp_audio = False
        if session_id:
            audio_info = tools_manager.get_session_data(session_id, "audio")
            if not audio_info:
                return {"status": "error", "message": "未找到音频数据，请先调用 video2audio"}
            
            audio_path = audio_info["audio_path"]
            work_dir = tools_manager.get_session_work_dir(session_id)
        elif audio_file_data and audio_filename:
            # 保存临时音频文件
            extension = Path(audio_filename).suffix.lower()
            if extension not in SUPPORTED_AUDIO_FORMATS:
                return {"status": "error", "message": f"不支持的音频格式: {extension}"}
            
            audio_path = tools_manager.save_temp_file(audio_file_data, extension, session_id)
            temp_audio = True
            work_dir = None
        else:
            return {"status": "error", "message": "需要提供 session_id 或 audio_file_data"}
        
        if not os.path.exists(audio_path):
            return {"status": "error", "message": "音频文件不存在"}
        
        # 获取音频信息
        audio_size = os.path.getsize(audio_path)
        logger.info(f"开始音频转录: {os.path.basename(audio_path)} ({audio_size/1024/1024:.2f} MB)")
        
        # 构建 Whisper 参数
        generate_kwargs = {}
        if language != "auto":
            generate_kwargs["language"] = language
        
        # 使用 Whisper 进行语音识别
        try:
            logger.info("正在进行语音识别...")
            if generate_kwargs:
                result = whisper_pipe(audio_path, generate_kwargs=generate_kwargs)
            else:
                result = whisper_pipe(audio_path)
            
            text_content = result.get("text", "").strip()
            
            if not text_content:
                return {
                    "status": "warning", 
                    "message": "未识别到任何文字内容", 
                    "text": "",
                    "session_id": session_id
                }
        
        except Exception as e:
            if temp_audio:
                tools_manager.cleanup_temp_files(audio_path)
            logger.error(f"语音识别失败: {e}")
            return {"status": "error", "message": f"语音识别失败: {str(e)}"}
        
        logger.info(f"转录完成: {len(text_content)} 字符")
        
        # 准备转录数据
        transcript_data = {
            "text": text_content,
            "language": language,
            "detected_language": result.get("language", "unknown"),
            "audio_path": audio_path,
            "audio_size": audio_size,
            "text_length": len(text_content),
            "word_count": len(text_content.split()) if text_content else 0
        }
        
        # 如果有会话，保存转录结果到服务器
        if session_id:
            tools_manager.save_processed_data(session_id, "transcript", transcript_data)
            
            # 额外保存纯文本文件（便于服务器端查看）
            if work_dir:
                transcript_file = work_dir / "transcript.txt"
                try:
                    with open(transcript_file, "w", encoding="utf-8") as f:
                        f.write(f"转录时间: {datetime.now().isoformat()}\n")
                        f.write(f"语言: {language}\n")
                        f.write(f"检测到的语言: {result.get('language', 'unknown')}\n")
                        f.write(f"字符数: {len(text_content)}\n")
                        f.write(f"词数: {len(text_content.split())}\n")
                        f.write("-" * 50 + "\n")
                        f.write(text_content)
                    logger.info(f"转录文本已保存: {transcript_file}")
                except Exception as e:
                    logger.error(f"保存转录文件失败: {e}")
        
        # 如果是临时文件，清理
        if temp_audio:
            tools_manager.cleanup_temp_files(audio_path)
        
        # 返回给客户端的完整数据
        return {
            "status": "success",
            "session_id": session_id,
            "text": text_content,  # 完整文本返回给客户端
            "language": language,
            "detected_language": result.get("language", "unknown"),
            "length": len(text_content),
            "word_count": len(text_content.split()) if text_content else 0,
            "audio_info": {
                "size": audio_size,
                "size_mb": round(audio_size / (1024*1024), 3),
                "path": audio_path  # 服务器端路径信息
            },
            "server_info": {
                "transcript_saved_to": str(work_dir / "transcript.txt") if work_dir else None,
                "processing_completed_at": datetime.now().isoformat()
            }
        }
    
    except Exception as e:
        logger.error(f"音频转文字时出错: {e}")
        return {"status": "error", "message": f"音频转文字时出错: {str(e)}"}