import asyncio
import logging
import base64
import os
from pathlib import Path
from fastmcp import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RemoteMCPClient:
    def __init__(self, server_ip: str, server_port: int = 8000):
        # 远程服务器的MCP URL
        self.mcp_url = f"http://{server_ip}:{server_port}/mcp/"
        self.client = Client(self.mcp_url)
        logger.info(f"连接到远程MCP服务器: {self.mcp_url}")

    async def initialize(self):
        logger.info("发送初始化请求到远程服务器...")
        await self.client.initialize()
        logger.info("远程MCP初始化成功")

    async def list_tools(self):
        logger.info("获取远程工具列表...")
        tools = await self.client.list_tools()
        logger.info(f"远程服务器可用工具数量: {len(tools)}")
        for tool in tools:
            logger.info(f"- {tool.name}: {tool.description or '无描述'}")
        return tools

    async def call_tool(self, tool_name: str, arguments: dict):
        logger.info(f"调用远程工具: {tool_name}")
        logger.debug(f"参数: {arguments}")
        try:
            result = await self.client.call_tool(tool_name, arguments)
            logger.info(f"远程工具调用成功: {tool_name}")
            if tool_name not in ["video2pictures_tool", "video2audio_tool", "picturepicker_tool"]:
                logger.info(f"结果: {result}")
            return result
        except Exception as e:
            logger.error(f"远程工具调用失败: {tool_name}, 错误: {e}")
            raise

    async def test_system_stats(self):
        """测试获取远程系统状态"""
        logger.info("=== 测试远程系统状态 ===")
        return await self.call_tool("get_system_stats_tool", {})

    async def test_video_upload(self, video_path: str = None):
        """测试上传视频到远程服务器"""
        if video_path and os.path.exists(video_path):
            logger.info(f"上传本地视频文件到远程服务器: {video_path}")
            with open(video_path, "rb") as f:
                video_data = base64.b64encode(f.read()).decode("utf-8")
            filename = Path(video_path).name
            
            # 显示文件大小信息
            file_size_mb = len(video_data.encode()) / (1024 * 1024) * 3/4  # base64编码大约增加33%
            logger.info(f"文件大小: {file_size_mb:.2f} MB")
            
        else:
            logger.error("指定的视频文件不存在！")
            return None

        return await self.call_tool("upload_video_tool", {
            "video_file_data": video_data,
            "filename": filename
        })

    async def run_comprehensive_test(self, video_path: str = None):
        """运行完整的远程测试流程"""
        logger.info("=== 远程MCP Video Tools 综合测试 ===")

        async with self.client:
            logger.info("远程MCP客户端已初始化")

            # 列出远程工具
            tools = await self.list_tools()
            if not tools:
                logger.warning("远程服务器没有找到可用工具")
                return False

            # 获取远程系统状态
            await self.test_system_stats()

            # 上传视频到远程服务器
            upload_result = await self.test_video_upload(video_path)
            session_id = upload_result.data.get("session_id") if upload_result else None

            if not session_id:
                logger.error("上传失败或未获取到 session_id")
                return False

            logger.info(f"远程会话 ID: {session_id}")

            # 远程视频处理流程
            processing_steps = [
                ("video2pictures_tool", {
                    "session_id": session_id,
                    "max_frames": 1000
                }),
                ("picturepicker_tool", {
                    "session_id": session_id,
                    "max_keyframes": 8
                }),
                ("video2audio_tool", {
                    "session_id": session_id
                }),
                ("audio2text_tool", {
                    "session_id": session_id
                }),
                ("video2metadata_tool", {
                    "session_id": session_id
                })
            ]

            for step_name, params in processing_steps:
                try:
                    logger.info(f"执行远程处理步骤: {step_name}")
                    result = await self.call_tool(step_name, params)
                    
                    if result and result.data:
                        if step_name == "video2pictures_tool":
                            logger.info(f"远程帧提取成功: 总帧数 {result.data.get('total_frames')}, 提取 {result.data.get('extracted_frames')} 帧")
                        elif step_name == "picturepicker_tool":
                            logger.info(f"远程关键帧选择成功: {result.data.get('selected_keyframes')} 张")
                        elif step_name == "video2audio_tool":
                            logger.info(f"远程音频提取成功: {result.data.get('format')}, {result.data.get('size_mb')} MB")
                        elif step_name == "audio2text_tool":
                            logger.info(f"远程文本转录成功: {result.data.get('word_count')} 词")
                        elif step_name == "video2metadata_tool":
                            logger.info(f"远程元数据提取成功: {result.data.get('resolution')}, {result.data.get('duration_formatted')}")
                    else:
                        logger.error(f"远程处理步骤失败: {step_name}")
                        
                except Exception as e:
                    logger.error(f"远程处理步骤出错 {step_name}: {e}")

            # 获取最终会话信息
            logger.info("获取最终远程会话信息...")
            session_info = await self.call_tool("get_session_info_tool", {"session_id": session_id})
            if session_info and session_info.data:
                logger.info(f"远程处理完成，进度: {session_info.data.get('progress_percentage')}%")

        logger.info("=== 远程测试完成 ===")
        return True


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="远程MCP Video Tools 测试客户端")
    parser.add_argument("--server-ip", required=True, help="远程服务器IP地址")
    parser.add_argument("--server-port", type=int, default=8000, help="远程服务器端口")
    parser.add_argument("--video", help="测试视频文件路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    client = RemoteMCPClient(args.server_ip, args.server_port)

    try:
        await client.run_comprehensive_test(args.video)
        logger.info("所有远程测试通过！")
    except Exception as e:
        logger.error(f"远程测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    # 使用示例:
    # python remote_client.py --server-ip 你的服务器IP --video test_video.mp4
    asyncio.run(main())
