#!/bin/bash

cd video-mcp-server

if [ -f server.pid ]; then
    PID=$(cat server.pid)
    if ps -p $PID > /dev/null; then
        echo "正在停止服务器进程 $PID..."
        kill $PID
        
        # 等待进程结束
        sleep 2
        if ps -p $PID > /dev/null; then
            echo "强制结束进程..."
            kill -9 $PID
        fi
        
        echo "服务器已停止"
    else
        echo "服务器进程不存在"
    fi
    rm -f server.pid
else
    echo "未找到进程ID文件"
    # 尝试通过端口杀死进程
    echo "尝试通过端口8000查找进程..."
    PID=$(lsof -ti:8000)
    if [ ! -z "$PID" ]; then
        echo "发现进程 $PID 占用端口8000，正在结束..."
        kill $PID
        echo "进程已结束"
    else
        echo "未发现占用端口8000的进程"
    fi
fi
