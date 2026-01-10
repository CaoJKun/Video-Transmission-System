import asyncio
import logging
import base64
import os
from pathlib import Path
from fastmcp import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


    async def call_tool(self, tool_name: str, arguments: dict):
        logger.info(f"调用工具: {tool_name}")
        logger.debug(f"参数: {arguments}")
        result = await self.client.call_tool(tool_name, arguments)
        logger.info(f"工具调用成功: {tool_name}")
        if tool_name != "video2pictures_tool" and tool_name != "video2audio_tool" and tool_name != "picturepicker_tool":
            logger.info(f"结果: {result}")
        return result

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

            # 调用 video2pictures_tool
            logger.info("调用 video2pictures_tool 提取帧")
            pic_result = await self.call_tool("video2pictures_tool", {
                "session_id": session_id,
                "max_frames": 1000
            })

            if pic_result and pic_result.data:
                logger.info(f"帧提取成功，总帧数: {pic_result.data.get('total_frames')}, 实际提取: {pic_result.data.get('extracted_frames')}")
            else:
                logger.error("帧提取失败")

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
        await client.run_comprehensive_test(args.video)
        logger.info("所有测试通过！")
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
