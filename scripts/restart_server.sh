#!/bin/bash

cd video-mcp-server

echo "=== 重启MCP Video Server ==="
echo "时间: $(date)"

# 停止服务器
echo "1. 停止现有服务器..."
./scripts/stop_server.sh

# 等待服务完全停止
echo "2. 等待服务完全停止..."
sleep 3

# 清理可能残留的进程
PIDS=$(pgrep -f "python.*main.py")
if [ ! -z "$PIDS" ]; then
    echo "发现残留进程，强制终止..."
    kill -9 $PIDS 2>/dev/null
fi

# 启动服务器
echo "3. 启动服务器..."
./scripts/start_server.sh

echo "=== 重启完成 ==="
EOF
