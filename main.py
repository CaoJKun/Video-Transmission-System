import asyncio
import uvicorn
import websockets
import json
import sys
import os
import logging
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
import fastmcp
from fastmcp import FastMCP
from fastapi.middleware.cors import CORSMiddleware

# 导入工具模块
from tools.video_manager import VideoToolsManager, SessionCleanupManager
from tools.upload_video import upload_video
from tools.video2pictures import video2pictures
from tools.video2rekeyframes import video2rekeyframes
from tools.picturepicker import picturepicker
from tools.video2audio import video2audio
from tools.audio2text import audio2text
from tools.video2metadata import video2metadata
from tools.video2motion_tokens import video2motionTokens
from tools.video2token import video2token
from tools.token2video import token2video
from tools.text2audio import text2audio, text2audio_from_transcript
from tools.pictures2video import pictures2video, pictures2video_from_session
from tools.audio_combine_video import audio_combine_video
from tools.session_info import get_session_info, cleanup_session, get_system_stats
from tools.utils import validate_ffmpeg

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置FastMCP设置 - 远程服务器配置
fastmcp.settings.host = "0.0.0.0"  # 监听所有网络接口
fastmcp.settings.port = int(os.environ.get("PORT", 8000))
fastmcp.settings.log_level = "info"

# 创建 FastMCP 实例
mcp = FastMCP("video-tools")

# 创建工作目录 - 确保在服务器上有足够的存储空间
current_dir = Path.cwd()
WORK_DIR = current_dir / "video_processing_data"  # 更明确的目录名
WORK_DIR.mkdir(parents=True, exist_ok=True)

# 全局变量
whisper_pipe = None
tools_manager = None
cleanup_manager = None

def init_whisper_model():
    """初始化 Whisper 模型，支持重试和备选方案"""
    global whisper_pipe
    max_retries = 3
    model_name = "openai/whisper-base"

    # 设置环境变量
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    
    logger.info(f"尝试加载模型: {model_name}")
    for attempt in range(max_retries):
        try:
            import torch
            import multiprocessing
            from transformers import logging as transformers_logging
            transformers_logging.set_verbosity_error()  # 只显示错误信息，减少日志
            from transformers import pipeline

            whisper_pipe = pipeline(
                task="automatic-speech-recognition",
                model=model_name,
                device=-1,                # CPU
                chunk_length_s=20,        # 20s 一片
                stride_length_s=(4, 2),   # 前后跨
                batch_size=16,            # 视内存可调 8~32
                return_timestamps=False,  # 关闭时间戳大幅提速
                generate_kwargs={
                    "task": "transcribe", # 始终转录（非翻译）
                    "language": "zh",     # 若业务稳定中文，建议固定语言
                    "num_beams": 1,       # 禁用 beam search
                    "do_sample": False,
                    "temperature": 0.0,
                }
            )
            logger.info(f"Whisper 模型加载成功: {model_name}")
            return
        except Exception as e:
            logger.warning(f"模型 {model_name} 加载失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)  # 等待更长时间
    
    logger.error("所有Whisper模型加载失败, 语音转文字功能将不可用")
    whisper_pipe = None


# 注册所有工具
@mcp.tool
async def upload_video_tool(video_file_data: str, filename: str) -> dict:
    return await upload_video(video_file_data, filename, tools_manager)

@mcp.tool
async def video2pictures_tool(
    session_id: str = None,
    video_file_data: str = None,
    video_filename: str = None,
    num_keyframes: int = 48
) -> dict:
    return await video2pictures(session_id, video_file_data, video_filename, num_keyframes, tools_manager)

@mcp.tool
async def video2rekeyframes_tool(
    session_id: str = None,
    video_file_data: str = None,
    video_filename: str = None,
    step: int = 5
) -> dict:
    return await video2rekeyframes(session_id, video_file_data, video_filename, step, tools_manager)

@mcp.tool
async def picturepicker_tool(
    session_id: str = None,
    frames_data: list = None,
    max_keyframes: int = 8,
    selection_method: str = "uniform"
) -> dict:
    return await picturepicker(session_id, frames_data, max_keyframes, selection_method, tools_manager)

@mcp.tool
async def video2audio_tool(
    session_id: str = None,
    video_file_data: str = None,
    video_filename: str = None,
    audio_format: str = "wav"
) -> dict:
    return await video2audio(session_id, video_file_data, video_filename, audio_format, tools_manager)

@mcp.tool
async def audio2text_tool(
    session_id: str = None,
    audio_file_data: str = None,
    audio_filename: str = None,
    language: str = "auto"
) -> dict:
    return await audio2text(session_id, audio_file_data, audio_filename, language, tools_manager, whisper_pipe)

@mcp.tool
async def video2metadata_tool(
    session_id: str = None,
    video_file_data: str = None,
    video_filename: str = None
) -> dict:
    return await video2metadata(session_id, video_file_data, video_filename, tools_manager)

@mcp.tool
async def video2motionTokens_tool(
    session_id: str = None,
    video_file_data: str = None,
    video_filename: str = None,
    token_length: int = 256,
    num_tokens: int = 8
) -> dict:
    return await video2motionTokens(session_id, video_file_data, video_filename, token_length, num_tokens, tools_manager)

@mcp.tool
async def video2token_tool(
    session_id: str = None,
    video_file_data: str = None,
    video_filename: str = None,
    token_dim: int = 256,
    chunk_size: int = 8,
    target_height: int = None
) -> dict:
    return await video2token(session_id, video_file_data, video_filename, token_dim, chunk_size, target_height, tools_manager)

@mcp.tool
async def token2video_tool(
    session_id: str = None,
    token_data: dict = None,
    output_filename: str = None,
    target_height: int = None
) -> dict:
    logger.info("🎬 开始token2video转换...")
    result = await token2video(session_id, token_data, output_filename, target_height, tools_manager)
    if result.get("status") == "success":
        logger.info("✅ token2video转换完成")
    else:
        logger.error(f"❌ token2video转换失败: {result.get('message', '未知错误')}")
    return result

@mcp.tool
def get_session_info_tool(session_id: str) -> dict:
    return get_session_info(session_id, tools_manager)

@mcp.tool
def cleanup_session_tool(session_id: str) -> dict:
    return cleanup_session(session_id, tools_manager)

@mcp.tool
async def text2audio_tool(
    session_id: str = None,
    text: str = None,
    reference_audio_path: str = None,
    reference_audio_data: str = None,
    output_filename: str = None,
    language: str = "zh"
) -> dict:
    return await text2audio(session_id, text, reference_audio_path, reference_audio_data, output_filename, language, None, tools_manager)

@mcp.tool
async def text2audio_from_transcript_tool(
    session_id: str,
    output_filename: str = None,
    language: str = "zh"
) -> dict:
    return await text2audio_from_transcript(session_id, output_filename, language, tools_manager)

@mcp.tool
async def pictures2video_tool(
    session_id: str = None,
    frames_data: list = None,
    output_filename: str = None,
    fps: float = 30.0,
    codec: str = "mp4v",
    quality: int = 90
) -> dict:
    return await pictures2video(session_id, frames_data, output_filename, fps, codec, quality, tools_manager)

@mcp.tool
async def pictures2video_from_session_tool(
    session_id: str,
    output_filename: str = None
) -> dict:
    return await pictures2video_from_session(session_id, output_filename, tools_manager)

@mcp.tool
async def audio_combine_video_tool(
    session_id: str = None,
    audio_file_path: str = None,
    video_file_path: str = None,
    output_filename: str = "combined_video.mp4"
) -> dict:
    return await audio_combine_video(session_id, audio_file_path, video_file_path, output_filename, tools_manager)

@mcp.tool
def get_system_stats_tool() -> dict:
    return get_system_stats(tools_manager, whisper_pipe, validate_ffmpeg, WORK_DIR)

def signal_handler(signum, frame):
    """信号处理器"""
    logger.info("\n收到关闭信号, 正在清理...")
    
    # 停止清理任务
    if cleanup_manager:
        cleanup_manager.stop()
    
    # 清理所有会话
    if tools_manager:
        session_ids = list(tools_manager.sessions.keys())
        for session_id in session_ids:
            tools_manager.cleanup_session(session_id)
    
    logger.info("清理完成, 退出程序")
    sys.exit(0)

def main():
    """主函数"""
    global tools_manager, cleanup_manager
    
    # 初始化管理器
    tools_manager = VideoToolsManager(work_dir_path=str(WORK_DIR))
    cleanup_manager = SessionCleanupManager(tools_manager)
    
    # 异步初始化 Whisper 模型
    threading.Thread(target=init_whisper_model, daemon=True).start()
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"Starting Video Tools MCP Server on {fastmcp.settings.host}:{fastmcp.settings.port}")
    logger.info("Available tools:")
    logger.info("- upload_video: 上传视频文件并创建会话")
    logger.info("- video2pictures: 将视频转换为图片帧")
    logger.info("- picturepicker: 从帧中选择关键帧")
    logger.info("- video2audio: 将视频转换为音频")
    logger.info("- audio2text: 将音频转换为文字")
    logger.info("- video2metadata: 获取视频元数据")
    logger.info("- video2motionTokens: 生成运动令牌")
    logger.info("- video2token: 将视频转换为高质量token序列（支持原视频分辨率或指定分辨率）")
    logger.info("- token2video: 将token序列转换回高质量视频（支持原视频分辨率或指定分辨率）")
    logger.info("- text2audio: 使用Coqui TTS将文本转换为音频，支持声音克隆")
    logger.info("- text2audio_from_transcript: 从转录文本生成音频（声音克隆）")
    logger.info("- pictures2video: 使用OpenCV将图片帧转换为视频")
    logger.info("- pictures2video_from_session: 从会话帧数据生成视频")
    logger.info("- get_session_info: 获取会话信息和状态")
    logger.info("- cleanup_session: 手动清理会话")
    logger.info("- get_system_stats: 获取系统统计信息")
    
    # 检查依赖
    logger.info(f"FFmpeg状态: {'可用' if validate_ffmpeg() else '不可用'}")
    logger.info(f"工作目录: {WORK_DIR}")

    # 调用 mcp.run()
    try:
        mcp.run(transport="streamable-http")
    except Exception as e:
        logger.error(f"服务器运行时出错: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("服务器已停止")
    except Exception as e:
        logger.error(f"服务器启动失败: {e}")
        sys.exit(1)
