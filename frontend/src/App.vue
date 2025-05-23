<template>
  <div class="d-flex layout-wrapper">
    <!-- 左侧导航栏（根据 sidebarVisible 控制） -->
    <Sidebar v-if="sidebarVisible" @close="sidebarVisible = false" />

    <!-- 右侧主区域 -->
    <div class="main-panel">
      <Topbar @toggleSidebar="toggleSidebar" />

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

const sidebarVisible = ref(true)

function toggleSidebar() {
  sidebarVisible.value = !sidebarVisible.value
}
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
  margin-top: 56px;
  overflow-y: auto;
  background-color: #f8f9fa;
}
</style>
