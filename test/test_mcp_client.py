import asyncio
import logging
import base64
import os
from pathlib import Path
from fastmcp import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 降低 MCP 客户端库的日志级别，避免显示资源清理时的正常错误
logging.getLogger("mcp.client.streamable_http").setLevel(logging.WARNING)


class MCPClient:
    def __init__(self, mcp_url: str):
        self.client = Client(mcp_url)

    async def initialize(self):
        logger.info("发送初始化请求...")
        await self.client.initialize()
        logger.info("MCP初始化成功")

    async def list_tools(self):
        logger.info("获取工具列表...")
        tools = await self.client.list_tools()
        logger.info(f"可用工具数量: {len(tools)}")
        for tool in tools:
            logger.info(f"- {tool.name}: {tool.description or '无描述'}")
        return tools


    async def call_tool(self, tool_name: str, arguments: dict, timeout: int = 60):
        logger.info(f"调用工具: {tool_name}")
        logger.debug(f"参数: {arguments}")
        
        try:
            # 为特定工具设置更长的超时时间
            if "text2audio" in tool_name:
                timeout = 120  # TTS工具需要更长时间
            elif "video2token" in tool_name:
                timeout = 180  # video2token工具需要3分钟
            
            result = await asyncio.wait_for(
                self.client.call_tool(tool_name, arguments),
                timeout=timeout
            )
            logger.info(f"工具调用成功: {tool_name}")
            if tool_name != "video2pictures_tool" and tool_name != "video2audio_tool" and tool_name != "picturepicker_tool" and tool_name != "video2token_tool":
                logger.info(f"结果: {result}")
            return result
        except asyncio.TimeoutError:
            logger.error(f"工具 {tool_name} 调用超时 ({timeout}秒)")
            return None
        except Exception as e:
            logger.error(f"工具 {tool_name} 调用失败: {e}")
            return None

    async def test_video_upload(self, video_path: str = None):
        if video_path and os.path.exists(video_path):
            logger.info(f"使用真实视频文件: {video_path}")
            with open(video_path, "rb") as f:
                video_data = base64.b64encode(f.read()).decode("utf-8")
            filename = Path(video_path).name
        else:
            logger.error("指定的视频文件不存在！")
            return None

        return await self.call_tool("upload_video_tool", {
            "video_file_data": video_data,
            "filename": filename
        })

    async def test_system_stats(self):
        return await self.call_tool("get_system_stats_tool", {})

    async def run_comprehensive_test(self, video_path: str = None):
        logger.info("=== MCP Video Tools 综合测试 ===")

        try:
            async with self.client:
                logger.info("MCP已初始化完毕")

                tools = await self.list_tools()
                if not tools:
                    logger.warning("没有找到可用工具")

                # 获取系统状态
                await self.test_system_stats()

                # 上传真实或模拟视频
                upload_result = await self.test_video_upload(video_path)
                session_id = upload_result.data.get("session_id") if upload_result else None

                if not session_id:
                    logger.error("上传失败或未获取到 session_id")
                    return False

                logger.info(f"会话 ID: {session_id}")

                # 调用 video2pictures_tool (自适应关键帧提取)
                logger.info("调用 video2pictures_tool 自适应提取关键帧")
                pic_result = await self.call_tool("video2pictures_tool", {
                    "session_id": session_id,
                    "num_keyframes": 48
                })

                if pic_result and pic_result.data:
                    logger.info(f"关键帧提取成功，总帧数: {pic_result.data.get('total_frames')}, 实际提取: {pic_result.data.get('extracted_keyframes')}")
                else:
                    logger.error("关键帧提取失败")

                # 调用 picturepicker_tool
                logger.info("调用 picturepicker_tool 工具提取关键帧")
                picker_result = await self.call_tool("picturepicker_tool", {
                    "session_id": session_id,
                    "max_keyframes": 16
                })
                if picker_result and picker_result.data:
                    logger.info(f"picker提取成功, 总帧数: {picker_result.data.get('total_frames')}, 实际提取: {picker_result.data.get('selected_keyframes')}")
                else:
                    logger.error("picker帧提取失败")

                # 调用 video2audio_tool
                logger.info("调用 video2audio_tool 工具提取音频")
                audio_result = await self.call_tool("video2audio_tool", {
                    "session_id": session_id
                })
                if audio_result and audio_result.data:
                    logger.info(f"audio提取成功, 音频文件: {audio_result.data.get('audio_filename')}, size: {audio_result.data.get('size')}")
                else:
                    logger.error("音频提取失败")

                
                # 调用 audio2text_tool
                logger.info("调用 audio2text_tool 工具提取文本")
                txt_result = await self.call_tool("audio2text_tool", {
                    "session_id": session_id
                })
                if txt_result and txt_result.data:
                    logger.info(f"text提取成功, 文本文件: {txt_result.data.get('text')}, 文本长度: {txt_result.data.get('length')}")
                else:
                    logger.error("文本提取失败")

                # 调用 video2metadata_tool
                logger.info("调用 video2metadata_tool 提取元数据")
                meta_result = await self.call_tool("video2metadata_tool", {
                    "session_id": session_id
                })
                if meta_result and meta_result.data:
                    logger.info(f"metadata提取成功")
                else:
                    logger.error("元数据提取失败")

                # 调用 video2token_tool
                logger.info("调用 video2token_tool 将视频转换为高质量token（480p）")
                token_result = await self.call_tool("video2token_tool", {
                    "session_id": session_id,
                    "token_dim": 256,
                    "chunk_size": 8,
                    "target_height": 480
                })
                if token_result and token_result.data and token_result.data.get("status") == "success":
                    token_data = token_result.data.get("video_tokens", {})
                    tokens_shape = token_data.get("tokens_shape", [])
                    video_info = token_data.get("video_info", {})
                    logger.info(f"视频转token成功，生成 {tokens_shape[0] if tokens_shape else 0} 个token")
                    logger.info(f"视频信息: 帧率={video_info.get('fps', 'N/A')}, 分辨率={video_info.get('width', 'N/A')}x{video_info.get('height', 'N/A')}")
                else:
                    err_msg = token_result.data.get("message") if (token_result and token_result.data) else "未知错误"
                    logger.error(f"视频转token失败: {err_msg}")
                    return False

                # 调用 token2video_tool
                logger.info("调用 token2video_tool 将token转换回高质量视频（480p）")
                video_result = await self.call_tool("token2video_tool", {
                    "session_id": session_id,
                    "output_filename": "decoded_test_video_480p.mp4",
                    "target_height": 480
                })
                if video_result and video_result.data and video_result.data.get("status") == "success":
                    output_path = video_result.data.get("output_path", "")
                    video_info = video_result.data.get("video_info", {})
                    logger.info(f"token转视频成功，输出路径: {output_path}")
                    logger.info(f"还原视频信息: 帧数={video_info.get('frames_count', 'N/A')}, 时长={video_info.get('duration', 'N/A')}秒")
                else:
                    err_msg = video_result.data.get("message") if (video_result and video_result.data) else "未知错误"
                    logger.error(f"token转视频失败: {err_msg}")
                    return False

                # 调用 text2audio_from_transcript_tool
                logger.info("调用 text2audio_from_transcript_tool 从转录文本生成音频（声音克隆）")
                tts_result = await self.call_tool("text2audio_from_transcript_tool", {
                    "session_id": session_id,
                    "output_filename": "cloned_voice_output.wav",
                    "language": "zh"
                })
                if tts_result and tts_result.data and tts_result.data.get("status") == "success":
                    output_path = tts_result.data.get("output_path", "")
                    audio_info = tts_result.data.get("audio_info", {})
                    tts_info = tts_result.data.get("tts_info", {})
                    logger.info(f"文本转音频成功，输出路径: {output_path}")
                    logger.info(f"音频信息: 大小={audio_info.get('size_mb', 'N/A')}MB, 时长={audio_info.get('duration', 'N/A')}秒")
                    logger.info(f"TTS信息: 模型={tts_info.get('model', 'N/A')}, 声音克隆={tts_info.get('voice_cloning', False)}")
                else:
                    err_msg = tts_result.data.get("message") if (tts_result and tts_result.data) else "未知错误"
                    logger.error(f"文本转音频失败: {err_msg}")
                    return False

                # 调用 text2audio_tool (自定义文本)
                logger.info("调用 text2audio_tool 使用自定义文本生成音频")
                custom_tts_result = await self.call_tool("text2audio_tool", {
                    "session_id": session_id,
                    "text": "这是一个测试文本，用于验证Coqui TTS的声音克隆功能。",
                    "output_filename": "custom_text_output.wav",
                    "language": "zh"
                })
                if custom_tts_result and custom_tts_result.data and custom_tts_result.data.get("status") == "success":
                    output_path = custom_tts_result.data.get("output_path", "")
                    audio_info = custom_tts_result.data.get("audio_info", {})
                    logger.info(f"自定义文本转音频成功，输出路径: {output_path}")
                    logger.info(f"音频信息: 大小={audio_info.get('size_mb', 'N/A')}MB, 时长={audio_info.get('duration', 'N/A')}秒")
                else:
                    err_msg = custom_tts_result.data.get("message") if (custom_tts_result and custom_tts_result.data) else "未知错误"
                    logger.error(f"自定义文本转音频失败: {err_msg}")
                    return False

                # 调用 pictures2video_from_session_tool (从关键帧生成视频，带运动插帧)
                logger.info("调用 pictures2video_from_session_tool 从关键帧生成视频（运动插帧）")
                frames2video_result = await self.call_tool("pictures2video_from_session_tool", {
                    "session_id": session_id,
                    "output_filename": "frames_to_video_output.mp4"
                })
                if frames2video_result and frames2video_result.data and frames2video_result.data.get("status") == "success":
                    output_path = frames2video_result.data.get("output_path", "")
                    video_info = frames2video_result.data.get("video_info", {})
                    processing_info = frames2video_result.data.get("processing_info", {})
                    logger.info(f"关键帧转视频成功，输出路径: {output_path}")
                    logger.info(f"视频信息: 分辨率={video_info.get('width', 'N/A')}x{video_info.get('height', 'N/A')}, 帧数={video_info.get('frame_count', 'N/A')}, 时长={video_info.get('duration', 'N/A')}秒")
                    logger.info(f"处理信息: 输入关键帧数={processing_info.get('input_keyframes', 'N/A')}, 处理帧数={processing_info.get('processed_frames', 'N/A')}, 编码器={processing_info.get('codec', 'N/A')}, 插帧={processing_info.get('interpolation', False)}")
                else:
                    err_msg = frames2video_result.data.get("message") if (frames2video_result and frames2video_result.data) else "未知错误"
                    logger.error(f"关键帧转视频失败: {err_msg}")
                    return False

                # 调用 audio_combine_video_tool
                logger.info("调用 audio_combine_video_tool 将音频和视频结合")
                combine_result = await self.call_tool("audio_combine_video_tool", {
                    "session_id": session_id,
                    "output_filename": "combined_video.mp4"
                })
                if combine_result and combine_result.data and combine_result.data.get("status") == "success":
                    output_path = combine_result.data.get("output_file", "")
                    processing_info = combine_result.data.get("processing_info", {})
                    logger.info(f"音频视频结合成功，输出路径: {output_path}")
                    logger.info(f"处理信息: 原始视频时长={processing_info.get('original_video_duration', 'N/A')}秒, 原始音频时长={processing_info.get('original_audio_duration', 'N/A')}秒, 最终时长={processing_info.get('final_duration', 'N/A')}秒")
                else:
                    err_msg = combine_result.data.get("message") if (combine_result and combine_result.data) else "未知错误"
                    logger.error(f"音频视频结合失败: {err_msg}")
                    return False

        except Exception as e:
            # 忽略资源清理时的 ClosedResourceError（这是正常的异步清理行为）
            error_name = type(e).__name__
            if error_name == "ClosedResourceError":
                logger.debug(f"资源清理时的正常错误（可忽略）: {error_name}")
                # 如果测试已经完成，返回 True
                logger.info("=== 测试完成 ===")
                return True
            else:
                # 其他错误需要记录
                logger.error(f"测试过程中出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False

        logger.info("=== 测试完成 ===")
        return True


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="MCP Video Tools 测试客户端")
    parser.add_argument("--url", default="http://localhost:8000/mcp/", help="MCP服务器 URL（streamable-http）")
    parser.add_argument("--video", help="测试视频文件路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    client = MCPClient(args.url)

    try:
        success = await client.run_comprehensive_test(args.video)
        if success:
            logger.info("所有测试通过！")
        else:
            logger.error("测试未通过！")
    except Exception as e:
        # 忽略资源清理时的 ClosedResourceError（这是正常的异步清理行为）
        error_name = type(e).__name__
        if error_name == "ClosedResourceError":
            logger.debug(f"资源清理时的正常错误（可忽略）: {error_name}")
        else:
            logger.error(f"测试失败: {e}")
            import traceback
            logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
