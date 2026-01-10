# 视频工具 MCP 服务器

一个全面的模型上下文协议 (MCP) 服务器，用于视频处理，提供视频分析、帧提取、音频处理和使用 AI 模型进行转录的工具。

## 功能特性

- **视频上传与会话管理**: 安全的视频上传与隔离的处理会话
- **帧提取**: 从视频中提取帧，可自定义间隔和限制
- **关键帧选择**: 使用均匀或随机采样智能选择代表性帧
- **音频提取**: 将视频转换为 WAV/MP3 格式的音频
- **语音转文字**: 使用 OpenAI Whisper 模型转录音频
- **视频元数据**: 提取全面的视频信息（分辨率、时长、比特率等）
- **运动令牌**: 生成运动表示令牌（实验性功能）
- **自动清理**: 自动会话清理以管理服务器资源
- **远程访问**: 完整的远程服务器支持

## 系统要求

- **Python**: >= 3.10 (已在 Python 3.13.5 上测试)
- **FFmpeg**: 视频/音频处理必需
- **OpenCV**: 用于帧提取
- **系统**: 推荐 Linux/macOS

## 安装

### 1. 克隆并设置环境

```bash
git clone <your-repo>
cd video-mcp-server

# 创建虚拟环境
# python3 -m venv venv 
# ⬆️ 这条命令也可以使用，关键是python版本需要>=3.10
pip install virtualenv 
virtualenv venv --python=python3.13.5

# 激活虚拟环境
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 安装系统依赖

**Ubuntu/Debian:**
```bash
# 可以使用pip安装, 本质上是一样的, 安装到虚拟环境中即可
# 建议用pip，macOS和windows都可以
pip install ffmpeg
# sudo apt update
# sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**验证安装:**
```bash
ffmpeg -version
```

## 快速开始

### 本地开发

```bash
# 使脚本可执行
chmod +x scripts/*.sh

# 启动服务器
# 方法1
./scripts/start_server.sh

# 方法2
# python3 main.py

# 服务器将在 http://localhost:8000 运行
# MCP 端点: http://localhost:8000/mcp/

# 注：
# 使用start_server.sh脚本启动服务器后，需要在logs/server.log中查看服务器输出，或使用命令`tail -f logs/server.log`查看
# 使用python3 main.py直接启动服务器，服务器将输出到终端上
```

### 远程部署

远程服务器部署：

```bash
# 使脚本可执行
chmod +x scripts/*.sh

# 启动服务器（将绑定到 0.0.0.0:8000）
# 直接在终端`python3 main.py`也行
./scripts/start_server.sh
```

服务器将在 `http://10.37.81.240:8000/mcp/` 可访问

## 测试

### 本地测试

```bash
# 使用本地视频文件测试
python3 test/test_mcp_client.py --video test/test_video_a2t.mp4 --verbose

# 不使用视频文件测试（使用模拟数据）
# 不推荐
python3 test/test_mcp_client.py --verbose
```

### 远程测试

```bash
# 测试远程服务器
python3 test/test_remote_client.py --server-ip 10.37.81.240 --video test/test_video_a2t.mp4
```

## 配置

### 环境变量

- `PORT`: 服务器端口（默认：8000）
- `PYTHONPATH`: 包含项目目录

### 服务器设置

服务器自动配置：
- **主机**: `0.0.0.0`（接受远程连接）
- **传输**: `streamable-http`
- **CORS**: 启用跨域请求
- **文件大小限制**: 每次上传 500MB
- **会话 TTL**: 24 小时自动清理

## 服务器管理

### 启动服务器
```bash
./scripts/start_server.sh
```

### 停止服务器
```bash
./scripts/stop_server.sh
```

### 重启服务器
```bash
./scripts/restart_server.sh
```

### 查看日志
```bash
tail -f logs/server.log
```

### 监控状态
```bash
# 检查服务器是否运行
curl http://localhost:8000/mcp/

# 通过 MCP 工具获取系统统计
python3 -c "
import asyncio
from fastmcp import Client

async def check_stats():
    async with Client('http://localhost:8000/mcp/') as client:
        await client.initialize()
        result = await client.call_tool('get_system_stats_tool', {})
        print(result)

asyncio.run(check_stats())
"
```

## 项目结构

```
video-mcp-server/
├── main.py                         # 主 MCP 服务器入口点
├── requirements.txt                # Python 依赖
├── README.md                       # 本文件
├── server.pid                      # 服务器进程 ID（运行时生成）
├── scripts/
│   ├── start_server.sh             # 服务器启动脚本
│   ├── stop_server.sh              # 服务器停止脚本
│   ├── restart_server.sh           # 服务器重启脚本
│   └── download_whisper.sh         # 预下载 Whisper 模型
├── test/
│   ├── test_mcp_client.py          # 本地测试客户端
│   ├── test_remote_client.py       # 远程服务器测试客户端
│   └── test_video_a2t.mp4          # 示例测试视频
├── tools/
│   ├── video_manager.py            # 会话和文件管理
│   ├── upload_video.py             # 视频上传处理器
│   ├── video2pictures.py           # 帧提取
│   ├── picturepicker.py            # 关键帧选择
│   ├── video2audio.py              # 音频提取
│   ├── audio2text.py               # 语音转录
│   ├── video2metadata.py           # 元数据提取
│   ├── video2motion_tokens.py      # 运动分析
│   ├── session_info.py             # 会话管理
│   └── utils.py                    # 工具函数
├── logs/                           # 服务器日志目录（启动后自动生成）
│   └── server.log                  # 服务器运行日志
└── video_processing_data/          # 视频处理数据目录（处理后生成，服务器停止后自动删除）
    ├── <session-id-1>/             # 会话目录（以会话ID命名）
    │   ├── original.mp4            # 原始上传的视频文件
    │   ├── session_info.json       # 会话信息文件
    │   ├── frames/                 # 提取的帧目录
    │   │   ├── frame_0000.jpg
    │   │   ├── frame_0001.jpg
    │   │   └── ...
    │   ├── keyframes/              # 关键帧目录
    │   │   ├── keyframe_0000.jpg
    │   │   ├── keyframe_0001.jpg
    │   │   └── ...
    │   ├── audio/                  # 提取的音频目录
    │   │   └── audio.wav           # 提取的音频文件
    │   ├── temp/                   # 临时文件目录
    │   ├── exports/                # 导出文件目录
    │   ├── transcript.txt          # 语音转录结果
    │   ├── metadata.json           # 视频元数据
    │   └── motion_tokens.json      # 运动令牌数据
    ├── <session-id-2>/             # 另一个会话目录
    │   └── ...
    └── ...                         # 更多会话目录
``` 


## 可用工具

### 1. `upload_video`
上传视频并创建处理会话。

**参数:**
- `video_file_data` (str): Base64 编码的视频文件
- `filename` (str): 原始文件名

**返回值:**
- `session_id`: 唯一会话标识符
- `file_size_mb`: 文件大小（MB）

### 2. `video2pictures`
从视频中提取帧。

**参数:**
- `session_id` (str): 来自上传的会话 ID
- `max_frames` (int): 最大提取帧数（1-1000，默认：100）

**返回值:**
- `frames`: Base64 编码的帧图像数组
- `total_frames`: 视频总帧数
- `extracted_frames`: 提取的帧数

### 3. `picturepicker`
从提取的帧中选择关键帧。

**参数:**
- `session_id` (str): 会话 ID
- `max_keyframes` (int): 选择的关键帧数量（1-50，默认：8）
- `selection_method` (str): "uniform" 或 "random"（默认："uniform"）

**返回值:**
- `keyframes`: 选择的关键帧图像
- `selected_keyframes`: 选择的关键帧数量

### 4. `video2audio`
从视频中提取音频。

**参数:**
- `session_id` (str): 会话 ID
- `audio_format` (str): "wav" 或 "mp3"（默认："wav"）

**返回值:**
- `audio_data`: Base64 编码的音频文件
- `size_mb`: 音频文件大小

### 5. `audio2text`
使用 Whisper 将音频转录为文本。

**参数:**
- `session_id` (str): 会话 ID（来自 video2audio）
- `language` (str): 语言代码（"auto", "en", "zh", "ja" 等）

**返回值:**
- `text`: 转录的文本
- `word_count`: 词数
- `detected_language`: 自动检测的语言

### 6. `video2metadata`
提取视频元数据。

**参数:**
- `session_id` (str): 会话 ID

**返回值:**
- `resolution`: 视频分辨率（如 "1920x1080"）
- `duration_formatted`: HH:MM:SS 格式的时长
- `fps`: 每秒帧数
- `bitrate_kbps`: 视频比特率

### 7. `video2motionTokens`
生成运动表示令牌（实验性）。

**参数:**
- `session_id` (str): 会话 ID
- `token_length` (int): 令牌维度（64-1024，默认：256）
- `num_tokens` (int): 令牌数量（1-100，默认：8）

### 8. 会话管理工具

- `get_session_info`: 获取会话状态和进度
- `cleanup_session`: 手动清理会话文件
- `get_system_stats`: 获取服务器统计信息和健康状态

## 安全与限制

- **文件上传**: 每个视频最大 500MB
- **会话隔离**: 每次上传创建隔离的工作目录
- **自动清理**: 会话在 24 小时后自动过期
- **格式验证**: 仅接受支持的视频格式
- **资源监控**: 内置系统资源跟踪

## 故障排除

### 常见问题

**1. 找不到 FFmpeg**
```bash
# 安装 FFmpeg
sudo apt install ffmpeg  # Ubuntu/Debian
brew install ffmpeg       # macOS
```

**2. Whisper 模型下载失败**
```bash
# 使用中国镜像（如需要）
export HF_ENDPOINT=https://hf-mirror.com
python3 scripts/download_whisper.sh
```

**3. 端口已被占用**
```bash
# 查找使用 8000 端口的进程
lsof -ti:8000
# 终止进程
kill $(lsof -ti:8000)
```

**4. 脚本权限被拒绝**
```bash
chmod +x scripts/*.sh
```

### 调试模式

启用详细日志记录：
```bash
python3 test/test_mcp_client.py --verbose
```

检查服务器日志：
```bash
tail -f logs/server.log
```

## 注意事项

- **文件大小**: 大文件上传可能需要较长时间，建议在稳定网络环境下使用
- **Whisper 模型**: 首次使用时会自动下载，可能需要几分钟时间
- **磁盘空间**: 视频处理会生成大量临时文件，确保有足够磁盘空间
- **内存使用**: 处理大视频时可能占用较多内存，建议监控系统资源

## 技术支持

遇到问题时：
1. 查看上述故障排除部分
2. 查看 `logs/server.log` 中的服务器日志
3. 使用提供的测试客户端进行测试
4. 提交 issue 时请包含详细的错误信息

---

**愉快地处理视频吧!**