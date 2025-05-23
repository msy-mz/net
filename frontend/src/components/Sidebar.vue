<template>
  <aside class="sidebar">
    <nav class="nav-cards">
      <div
        v-for="item in navs"
        :key="item.name"
        class="nav-item"
        :class="{ active: $route.path === item.route }"
        @click="navigate(item.route)"
      >
        <i :class="item.icon"></i>
        <span class="label">{{ item.name }}</span>
      </div>
    </nav>
  </aside>
</template>

<script setup>
import { useRouter, useRoute } from 'vue-router'
const emit = defineEmits(['close'])

const router = useRouter()
const route = useRoute()

const navs = [
  { name: '上传文件', route: '/upload', icon: 'fas fa-upload' },
  { name: '实时监听', route: '/monitor', icon: 'fas fa-signal' },
  { name: '用户管理', route: '/user', icon: 'fas fa-user-cog' }
]

function navigate(routePath) {
  router.push(routePath)
  emit('close') // 始终触发关闭
}

</script>

<style scoped>
.sidebar {
  position: fixed;
  top: 56px;
  left: 0;
  width: 240px;
  height: calc(100vh - 56px);
  background-color: #ffffff;
  border-right: 1px solid #e0e0e0;
  padding: 20px 12px;
  z-index: 1000;
  box-shadow: 4px 0 20px rgba(0, 0, 0, 0.06);
  transition: transform 0.3s ease;
}

.nav-cards {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  background-color: #f8f9fa;
  border-radius: 10px;
  font-size: 15px;
  color: #444;
  cursor: pointer;
  transition: all 0.2s ease;
}
.nav-item:hover {
  background-color: #e3eefe;
}
.nav-item.active {
  background-color: #d0e4ff;
  color: #1a4db3;
  font-weight: bold;
}
</style>
