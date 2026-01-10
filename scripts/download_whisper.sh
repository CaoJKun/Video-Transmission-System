#!/bin/bash

# 预下载Whisper模型脚本
cd video-mcp-server
source venv/bin/activate

echo "开始下载Whisper模型..."

# 方法1: 直接下载模型
python3 -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用镜像站点
from transformers import pipeline
print('正在下载Whisper Base模型...')
try:
    whisper_pipe = pipeline('automatic-speech-recognition', model='openai/whisper-base', device=-1)
    print('Whisper模型下载成功')
except Exception as e:
    print(f'下载失败: {e}')
    print('尝试使用镜像站点...')
    try:
        # 使用国内镜像
        whisper_pipe = pipeline('automatic-speech-recognition', model='openai/whisper-base', device=-1)
        print('通过镜像下载成功')
    except Exception as e2:
        print(f'镜像下载失败: {e2}')
"

# 方法2: 手动下载模型文件（如果上面失败）
if [ $? -ne 0 ]; then
    echo "自动下载失败，尝试手动下载..."
    mkdir -p ~/.cache/huggingface/transformers
    
    # 下载whisper-small模型（更小，下载更快）
    echo "切换到whisper-small模型..."
    python3 -c "
from transformers import pipeline
try:
    whisper_pipe = pipeline('automatic-speech-recognition', model='openai/whisper-small', device=-1)
    print('Whisper-small模型准备就绪')
except Exception as e:
    print(f'whisper-small下载失败: {e}')
    "
fi

echo "模型下载脚本执行完成"
