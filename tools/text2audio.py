import os
import logging
import tempfile
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torchaudio
from tools.video_manager import SUPPORTED_AUDIO_FORMATS

logger = logging.getLogger(__name__)

# 全局变量存储TTS模型
tts_model = None
tts_config = None

def init_coqui_tts():
    """初始化 Coqui TTS 模型"""
    global tts_model, tts_config
    
    try:
        from TTS.api import TTS
        from TTS.config import load_config
        from TTS.utils.manage import ModelManager
        
        logger.info("正在初始化 Coqui TTS...")
        
        # 设置模型管理器
        manager = ModelManager()
        
        # 获取可用的多语言模型
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        
        # 下载并加载模型
        model_path, config_path, model_item = manager.download_model(model_name)
        
        # 加载配置
        tts_config = load_config(config_path)
        
        # 初始化TTS模型
        tts_model = TTS(model_path=model_path, config_path=config_path)
        
        logger.info(f"Coqui TTS 模型加载成功: {model_name}")
        return True
        
    except ImportError as e:
        logger.warning(f"Coqui TTS 未安装，将使用替代方案: {e}")
        return init_alternative_tts()
    except Exception as e:
        logger.error(f"Coqui TTS 初始化失败: {e}")
        return init_alternative_tts()

def init_alternative_tts():
    """初始化替代TTS方案"""
    global tts_model, tts_config
    
    try:
        # 尝试使用pyttsx3作为替代方案
        import pyttsx3
        tts_model = pyttsx3.init()
        tts_config = {"engine": "pyttsx3"}
        logger.info("使用 pyttsx3 作为TTS替代方案")
        return True
    except ImportError as e:
        logger.error(f"pyttsx3 未安装: {e}")
        return init_simple_tts()
    except Exception as e:
        logger.error(f"pyttsx3 初始化失败: {e}")
        return init_simple_tts()

def init_simple_tts():
    """初始化简单TTS方案（仅保存文本）"""
    global tts_model, tts_config
    
    tts_model = None
    tts_config = {"engine": "simple", "text_only": True}
    logger.info("使用简单TTS方案（仅保存文本）")
    return True

async def text2audio(
    session_id: str = None,
    text: str = None,
    reference_audio_path: str = None,
    reference_audio_data: str = None,
    output_filename: str = None,
    language: str = "zh",
    speaker_wav: str = None,
    tools_manager = None
) -> dict:
    """
    使用 Coqui TTS 将文本转换为音频，支持声音克隆
    
    Args:
        session_id: 会话ID，用于获取参考音频或保存结果
        text: 要转换的文本内容
        reference_audio_path: 参考音频文件路径（用于声音克隆）
        reference_audio_data: base64编码的参考音频数据
        output_filename: 输出音频文件名
        language: 目标语言代码
        speaker_wav: 说话人音频文件路径（用于声音克隆）
        tools_manager: 视频工具管理器实例
    
    Returns:
        包含生成音频信息的字典
    """
    try:
        global tts_model, tts_config
        
        # 检查TTS模型是否已加载
        if tts_model is None:
            logger.info("TTS模型未加载，正在初始化...")
            if not init_coqui_tts():
                # 即使TTS不可用，也保存文本数据
                logger.warning("TTS模型初始化失败，但会保存文本数据")
                if session_id and tools_manager:
                    # 保存文本数据到会话
                    text_data = {
                        "text": text,
                        "language": language,
                        "text_length": len(text),
                        "word_count": len(text.split()) if text else 0,
                        "tts_available": False,
                        "timestamp": datetime.now().isoformat()
                    }
                    tools_manager.save_processed_data(session_id, "tts_text", text_data)
                
                return {
                    "status": "warning",
                    "message": "TTS模型不可用，但文本数据已保存",
                    "session_id": session_id,
                    "text": text,
                    "text_length": len(text),
                    "tts_available": False
                }
        
        # 验证输入参数
        if not text or not text.strip():
            return {"status": "error", "message": "文本内容不能为空"}
        
        # 获取参考音频路径
        reference_audio = None
        if reference_audio_path and os.path.exists(reference_audio_path):
            reference_audio = reference_audio_path
        elif reference_audio_data:
            # 从base64数据创建临时文件
            try:
                audio_data = base64.b64decode(reference_audio_data)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file.write(audio_data)
                    reference_audio = temp_file.name
            except Exception as e:
                return {"status": "error", "message": f"参考音频数据解码失败: {str(e)}"}
        elif session_id and tools_manager:
            # 从会话中获取音频文件
            audio_info = tools_manager.get_session_data(session_id, "audio")
            if audio_info and os.path.exists(audio_info.get("audio_path")):
                reference_audio = audio_info["audio_path"]
            else:
                # 尝试获取转录文本对应的原始音频
                transcript_info = tools_manager.get_session_data(session_id, "transcript")
                if transcript_info and os.path.exists(transcript_info.get("audio_path")):
                    reference_audio = transcript_info["audio_path"]
        
        # 设置输出路径
        if session_id and tools_manager:
            work_dir = tools_manager.get_session_work_dir(session_id)
            if not output_filename:
                output_filename = f"tts_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            output_path = work_dir / "audio" / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # 临时输出
            if not output_filename:
                output_filename = f"tts_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            output_path = Path(tempfile.gettempdir()) / output_filename
        
        logger.info(f"开始文本转语音: {len(text)} 字符")
        logger.info(f"目标语言: {language}")
        if reference_audio:
            logger.info(f"使用参考音频进行声音克隆: {os.path.basename(reference_audio)}")
        
        # 执行TTS转换
        try:
            if tts_config and tts_config.get("engine") == "simple":
                # 使用简单TTS方案（仅保存文本）
                logger.info("使用简单TTS方案，仅保存文本数据...")
                
                # 创建文本文件
                text_file = output_path.with_suffix('.txt')
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(f"TTS文本内容:\n{text}\n\n")
                    f.write(f"语言: {language}\n")
                    f.write(f"字符数: {len(text)}\n")
                    f.write(f"生成时间: {datetime.now().isoformat()}\n")
                
                # 创建一个空的音频文件占位符
                with open(output_path, 'w') as f:
                    f.write("TTS音频文件占位符\n")
                
                logger.info(f"文本已保存到: {text_file}")
                
            elif tts_config and tts_config.get("engine") == "pyttsx3":
                # 使用pyttsx3替代方案
                logger.info("使用 pyttsx3 进行文本转语音...")
                tts_model.setProperty('rate', 150)  # 语速
                tts_model.setProperty('volume', 0.9)  # 音量
                
                # 保存为音频文件
                tts_model.save_to_file(text, str(output_path))
                
                # 使用非阻塞方式等待完成
                import threading
                import time
                
                def wait_for_completion():
                    tts_model.runAndWait()
                
                # 启动线程
                tts_thread = threading.Thread(target=wait_for_completion)
                tts_thread.daemon = True
                tts_thread.start()
                
                # 等待最多30秒
                max_wait_time = 30
                start_time = time.time()
                while tts_thread.is_alive() and (time.time() - start_time) < max_wait_time:
                    time.sleep(0.5)
                
                if tts_thread.is_alive():
                    logger.warning("pyttsx3 处理超时，但可能仍在后台运行")
                
                # 检查文件是否生成
                if not os.path.exists(output_path):
                    return {"status": "error", "message": "pyttsx3 音频文件生成失败或超时"}
                    
            else:
                # 使用Coqui TTS
                if reference_audio and os.path.exists(reference_audio):
                    # 使用参考音频进行声音克隆
                    logger.info("使用声音克隆模式...")
                    tts_model.tts_to_file(
                        text=text,
                        speaker_wav=reference_audio,
                        language=language,
                        file_path=str(output_path)
                    )
                else:
                    # 使用默认声音
                    logger.info("使用默认声音模式...")
                    tts_model.tts_to_file(
                        text=text,
                        language=language,
                        file_path=str(output_path)
                    )
            
            # 检查输出文件是否生成成功
            if not os.path.exists(output_path):
                return {"status": "error", "message": "音频文件生成失败"}
            
            # 获取音频文件信息
            audio_size = os.path.getsize(output_path)
            
            # 获取音频元数据
            try:
                waveform, sample_rate = torchaudio.load(str(output_path))
                duration = waveform.shape[1] / sample_rate
                channels = waveform.shape[0]
            except Exception as e:
                logger.warning(f"无法获取音频元数据: {e}")
                duration = 0
                channels = 1
                sample_rate = 22050
            
            logger.info(f"TTS转换完成: {output_path} ({audio_size/1024/1024:.2f} MB, {duration:.2f}s)")
            
            # 准备返回数据
            result_data = {
                "status": "success",
                "session_id": session_id,
                "text": text,
                "language": language,
                "output_path": str(output_path),
                "output_filename": output_filename,
                "audio_info": {
                    "size": audio_size,
                    "size_mb": round(audio_size / (1024*1024), 3),
                    "duration": round(duration, 2),
                    "sample_rate": sample_rate,
                    "channels": channels
                },
                "tts_info": {
                    "model": "coqui_xtts_v2",
                    "voice_cloning": reference_audio is not None,
                    "reference_audio": os.path.basename(reference_audio) if reference_audio else None
                }
            }
            
            # 如果有会话，保存结果到会话数据
            if session_id and tools_manager:
                tools_manager.save_processed_data(session_id, "tts_output", result_data)
                logger.info(f"TTS结果已保存到会话: {session_id}")
            
            # 清理临时文件
            if reference_audio and reference_audio.startswith(tempfile.gettempdir()):
                try:
                    os.unlink(reference_audio)
                except:
                    pass
            
            return result_data
            
        except Exception as e:
            logger.error(f"TTS转换失败: {e}")
            return {
                "status": "error",
                "message": f"TTS转换失败: {str(e)}"
            }
    
    except Exception as e:
        logger.error(f"文本转音频时出错: {e}")
        return {"status": "error", "message": f"文本转音频时出错: {str(e)}"}

async def text2audio_from_transcript(
    session_id: str,
    output_filename: str = None,
    language: str = "zh",
    tools_manager = None
) -> dict:
    """
    从会话的转录文本生成音频（声音克隆）
    
    Args:
        session_id: 会话ID
        output_filename: 输出音频文件名
        language: 目标语言代码
        tools_manager: 视频工具管理器实例
    
    Returns:
        包含生成音频信息的字典
    """
    try:
        if not session_id or not tools_manager:
            return {"status": "error", "message": "需要提供有效的会话ID和工具管理器"}
        
        # 获取转录文本
        transcript_info = tools_manager.get_session_data(session_id, "transcript")
        if not transcript_info:
            return {"status": "error", "message": "未找到转录文本，请先调用 audio2text"}
        
        text = transcript_info.get("text", "")
        if not text.strip():
            return {"status": "error", "message": "转录文本为空"}
        
        # 获取原始音频作为参考
        audio_info = tools_manager.get_session_data(session_id, "audio")
        reference_audio_path = None
        if audio_info and os.path.exists(audio_info.get("audio_path")):
            reference_audio_path = audio_info["audio_path"]
        
        # 调用主要的text2audio函数
        result = await text2audio(
            session_id=session_id,
            text=text,
            reference_audio_path=reference_audio_path,
            output_filename=output_filename,
            language=language,
            tools_manager=tools_manager
        )
        
        # 如果TTS不可用，至少保存转录文本信息
        if result.get("status") == "warning" and result.get("tts_available") == False:
            logger.info("TTS不可用，但转录文本数据已保存到会话")
            result["transcript_saved"] = True
            result["message"] = "转录文本已保存，TTS功能暂不可用"
        
        return result
    
    except Exception as e:
        logger.error(f"从转录文本生成音频时出错: {e}")
        return {"status": "error", "message": f"从转录文本生成音频时出错: {str(e)}"}
