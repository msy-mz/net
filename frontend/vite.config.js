import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'

export default defineConfig({
  base: './',
  plugins: [vue()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src')
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
  },
  build: {
    chunkSizeWarningLimit: 700,  // 默认是 500KB，这里稍微放宽
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules')) {
            if (id.includes('chart.js')) return 'vendor-chart'
            if (id.includes('fontawesome')) return 'vendor-fonts'
            if (id.includes('bootstrap')) return 'vendor-bootstrap'
            return 'vendor'  // 其余统一打包到 vendor
          }
        }
      }
    }
  }
})
