// src/router/index.js
import { createRouter, createWebHistory } from 'vue-router'
import Upload from '@/views/Upload.vue'
import Monitor from '@/views/Monitor.vue'
import User from '@/views/User.vue'

const routes = [
  { path: '/', redirect: '/upload' },          //  启动页重定向到 /upload
  { path: '/upload', component: Upload },      //  显式上传页
  { path: '/monitor', component: Monitor },    //  添加监听页，防止空白
  { path: '/user', component: User }           //  其他页面
]

export default createRouter({
  history: createWebHistory(),
  routes
})
