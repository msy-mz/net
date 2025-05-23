// Filename: main.js
// Path: src/main.js
// Description: 应用入口，挂载路由和样式
// Author: msy
// Date: 2025

import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import './assets/main.css'
import '@fortawesome/fontawesome-free/css/all.min.css'
import 'bootstrap/dist/css/bootstrap.min.css'  


const app = createApp(App)
app.use(router)
app.mount('#app')
