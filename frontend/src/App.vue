<!-- Filename: App.vue -->
<!-- Path: frontend-vue/src/App.vue -->
<!-- Description: 应用根组件，提供左侧菜单栏 + 顶部栏 + 页面视图区 -->
<!-- Author: msy -->
<!-- Date: 2025 -->

<template>
  <div class="d-flex" style="height: 100vh; overflow: hidden;">
    <Sidebar />
    <div class="flex-grow-1 d-flex flex-column" style="overflow-y: auto;">
      <Topbar />
      <main class="flex-grow-1 p-4 bg-light" style="overflow-y: auto;">
        <router-view />
      </main>
    </div>
  </div>
</template>

<script setup>
import Sidebar from './components/Sidebar.vue'
import Topbar from './components/Topbar.vue'
</script>
