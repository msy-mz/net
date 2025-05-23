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
import Topbar from './components/TopBar.vue' 
</script>

