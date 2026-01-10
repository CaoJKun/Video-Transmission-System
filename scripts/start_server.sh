#!/bin/bash

# 进入项目目录
cd video-mcp-server

# 激活虚拟环境
source venv/bin/activate

# 设置环境变量
export PORT=8000
export PYTHONPATH=video-mcp-server:$PYTHONPATH

# 创建日志目录
mkdir -p logs

# 启动服务器，输出日志到文件
echo "启动Video MCP Server..."
echo "时间: $(date)"
echo "工作目录: $(pwd)"
echo "Python版本: $(python --version)"

# 启动服务
nohup python main.py > logs/server.log 2>&1 &

# 获取进程ID
PID=$!
echo "服务器进程ID: $PID"
echo $PID > server.pid

# 等待几秒检查服务是否正常启动
sleep 3
if ps -p $PID > /dev/null; then
    echo "服务器启动成功！"
    echo "日志文件: logs/server.log"
    echo "查看实时日志: tail -f logs/server.log"
    echo "停止服务器: ./stop_server.sh"
else
    echo "服务器启动失败，请检查日志"
    cat logs/server.log
fi

