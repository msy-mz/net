<template>
  <div class="layout">
    <aside class="sidebar-wrapper">
      <Sidebar />
    </aside>
    <div class="main-wrapper">
      <main class="content">
        <router-view />
      </main>
    </div>
  </div>
</template>

<script setup>
import Sidebar from '../components/Sidebar.vue'
import SystemTitle from '../components/SystemTitle.vue'
</script>

<template>
  <div class="layout">
    <aside class="sidebar-wrapper">
      <Sidebar />
    </aside>
    <div class="main-wrapper">
      <SystemTitle />
      <main class="content">
        <router-view />
      </main>
    </div>
  </div>

  
</template>


<style scoped>
.layout {
  display: flex;
  height: 100vh;
  background-color: #f5f7fa;
}
.sidebar-wrapper {
  flex-shrink: 0;
  background-color: #ffffff;
  border-right: 1px solid #e5e5e5;
  box-shadow: 2px 0 6px rgba(0, 0, 0, 0.03);
}
.main-wrapper {
  flex: 1;
  display: flex;
  flex-direction: column;
}
.content {
  flex: 1;
  padding: 36px;
  overflow-y: auto;
  background-color: #f7f9fb;
}
</style>

