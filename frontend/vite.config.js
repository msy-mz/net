// Filename: vite.config.js
// Path: frontend/vite.config.js
// Description: Vite 构建配置，设置路径别名和代理，适配 Flask 部署
// Author: msy
// Date: 2025

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'  // 必须导入 path 模块

export default defineConfig({
  base: './',  // 关键配置，确保构建资源路径为相对路径，适配 Flask 静态托管
  plugins: [vue()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src')  // 配置 @ 指向 src 目录
    }
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8888',
        changeOrigin: true,
        rewrite: path => path.replace(/^\/api/, '/api')
      }
    }
  }
})
