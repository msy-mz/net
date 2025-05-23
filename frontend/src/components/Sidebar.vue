<!-- Filename: Sidebar.vue -->
<!-- Path: frontend-vue/src/components/Sidebar.vue -->
<!-- Description: 左侧模块导航栏 -->
<!-- Author: msy -->
<!-- Date: 2025 -->

<template>
  <div :class="['sidebar bg-dark text-white p-3', { collapsed }]" style="transition: all 0.3s;" >
    <h5 class="mb-4 d-flex justify-content-between align-items-center">
      FT-IDNet++
      <i class="fas fa-bars d-md-none" @click="collapsed = !collapsed" style="cursor: pointer;"></i>
    </h5>
    <ul class="nav flex-column" v-show="!collapsed || isDesktop">
      <li v-for="item in navs" :key="item.name" class="nav-item">
        <a class="nav-link text-white" href="#" @click.prevent="navigate(item.route)">
          <i :class="item.icon" class="me-2"></i>{{ item.name }}
        </a>
      </li>
    </ul>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const collapsed = ref(false)
const isDesktop = ref(true)

onMounted(() => {
  isDesktop.value = window.innerWidth >= 768
  collapsed.value = !isDesktop.value
  window.addEventListener('resize', () => {
    isDesktop.value = window.innerWidth >= 768
  })
})


const navs = [
  { name: '系统总览', route: '/', icon: 'fas fa-tachometer-alt' },
  { name: '实时监听', route: '/monitor', icon: 'fas fa-wifi' },
  { name: '上传分析', route: '/upload', icon: 'fas fa-upload' },
  { name: '可视化分析', route: '/visual', icon: 'fas fa-chart-line' },
  { name: '监听日志', route: '/log', icon: 'fas fa-list' },
  { name: 'NIST 测试', route: '/nist-test', icon: 'fas fa-vial' }
]

const navigate = (route) => {
  if (!isDesktop.value) {
    collapsed.value = true
  }
  router.push(route)
}

</script>

<style scoped>
.sidebar {
  width: 220px;
}
.sidebar.collapsed {
  width: 0;
  padding: 0;
  overflow: hidden;
}
</style>