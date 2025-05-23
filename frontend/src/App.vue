<template>
  <div class="d-flex layout-wrapper">
    <!-- 侧边栏 -->
    <Sidebar v-model:collapsed="sidebarCollapsed" />

    <!-- 右侧主区域 -->
    <div class="main-panel">
      <!-- 顶部栏固定 -->
      <Topbar @toggleSidebar="sidebarCollapsed = !sidebarCollapsed" />

      <!-- 主内容：加 margin-top 避免被顶栏遮挡 -->
      <main class="main-content">
        <keep-alive include="Monitor">
          <router-view />
        </keep-alive>
      </main>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import Sidebar from './components/Sidebar.vue'
import Topbar from './components/TopBar.vue'

const sidebarCollapsed = ref(false)
</script>

<style scoped>
.layout-wrapper {
  height: 100vh;
  overflow: hidden;
}

.main-panel {
  flex-grow: 1;
  display: flex;
  flex-direction: column;
}

.main-content {
  flex-grow: 1;
  padding: 24px;
  margin-top: 56px; /* 顶栏高度 */
  overflow-y: auto;
  background-color: #f8f9fa;
}
</style>
