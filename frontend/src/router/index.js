import { createRouter, createWebHistory } from 'vue-router'
import Dashboard from '@/views/Dashboard.vue'
import Monitor from '@/views/Monitor.vue'
import Upload from '@/views/Upload.vue'
import Visual from '@/views/Visual.vue'
import Log from '@/views/Log.vue'
import NistTest from '@/views/NistTest.vue'
import User from '@/views/User.vue'  // 新增
import Payload from '@/views/Payload.vue'  // 路径请根据实际情况调整

const routes = [
  { path: '/', component: Dashboard },
  { path: '/monitor', component: Monitor },
  { path: '/upload', component: Upload },
  { path: '/visual', component: Visual },
  { path: '/log', component: Log },
  { path: '/nist-test', component: NistTest },
  { path: '/user', component: User }, // 新增
  { path: '/payload', name: 'Payload', component: Payload }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default createRouter({
  history: createWebHistory(),
  routes
})
