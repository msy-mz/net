<template>
  <div class="topbar fixed-top shadow-sm d-flex align-items-center px-3">
    <i class="fas fa-bars toggle-icon" @click="$emit('toggleSidebar')"></i>
    <h5 class="m-0 text-center flex-grow-1 topbar-title">基于 FT-IDNet++ 的网络流量监测系统</h5>
    <div style="width: 24px;"></div> <!-- 右侧占位防止标题偏左 -->
  </div>
</template>

<style scoped>
.topbar {
  height: 56px;
  background: #fff;
  z-index: 1030;
  width: 100%;
}

.toggle-icon {
  font-size: 20px;
  cursor: pointer;
  color: #333;
}

.topbar-title {
  font-size: 18px;
  font-weight: bold;
}
</style>
