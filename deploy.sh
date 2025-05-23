#!/bin/bash
# Filename: deploy.sh
# Path: net/deploy.sh
# Description: 一键部署脚本，打包前端并启动后端 Flask 服务
# Author: msy
# Date: 2025

echo " [1/3] 正在打包前端项目..."
cd frontend || exit
npm install > /dev/null
npm run build

echo " [2/3] 启动后端 Flask 服务..."
cd ../backend || exit

LOGFILE="../logs/backend_$(date +%Y%m%d_%H%M%S).log"
mkdir -p ../logs

nohup python app.py > "$LOGFILE" 2>&1 &
BACKEND_PID=$!

echo " 后端已启动，PID: $BACKEND_PID"
echo " 日志文件：$LOGFILE"
echo " 访问地址：http://localhost:8888/"
