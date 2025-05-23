import axios from 'axios'

// 创建 axios 实例
const request = axios.create({
  baseURL: '/api',        // 使用 Vite 代理，自动指向 http://localhost:8888
  timeout: 10000          // 超时时间（毫秒）
})

// 请求拦截器
request.interceptors.request.use(
  config => {
    return config
  },
  error => Promise.reject(error)
)

// 响应拦截器
request.interceptors.response.use(
  response => response.data,
  error => {
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

export default request
