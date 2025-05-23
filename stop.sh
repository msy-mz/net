#!/bin/bash
PIDS=$(lsof -ti:8888)
if [ -z "$PIDS" ]; then
  echo " 端口 8888 当前未被占用"
else
  echo "🔧 正在终止占用 8888 的进程: $PIDS"
  kill -9 $PIDS
  echo " 已释放端口 8888"
fi
