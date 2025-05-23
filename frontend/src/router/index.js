// Filename: index.js
// Path: src/router/index.js
// Description: Vue Router 路由配置
// Author: msy
// Date: 2025

import { createRouter, createWebHistory } from 'vue-router'
import Dashboard from '../views/Dashboard.vue'
import Monitor from '../views/Monitor.vue'
import Upload from '../views/Upload.vue'
import Visual from '../views/Visual.vue'
import Log from '../views/Log.vue'
import NistTest from '../views/NistTest.vue';

const routes = [
  { path: '/', name: 'Dashboard', component: Dashboard },
  { path: '/monitor', name: 'Monitor', component: Monitor },
  { path: '/upload', name: 'Upload', component: Upload },
  { path: '/visual', name: 'Visual', component: Visual },
  { path: '/log', name: 'Log', component: Log },
  { path: '/nist-test', name: 'NistTest', component: NistTest}
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
