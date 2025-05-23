<template>
  <aside ref="sidebarRef" :class="['sidebar', { collapsed }]">
    <nav class="nav-cards">
      <div
        v-for="item in navs"
        :key="item.name"
        class="nav-item"
        :class="{ active: $route.path === item.route }"
        @click="navigate(item.route)"
      >
        <i :class="item.icon"></i>
        <span v-show="!collapsed" class="label">{{ item.name }}</span>
      </div>
    </nav>
  </aside>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { useRouter, useRoute } from 'vue-router'

const props = defineProps({ collapsed: Boolean })
const emit = defineEmits(['update:collapsed'])

const router = useRouter()
const route = useRoute()
const sidebarRef = ref(null)

const collapsed = computed({
  get: () => props.collapsed,
  set: val => emit('update:collapsed', val)
})

const navs = [
  { name: '上传分析', route: '/upload', icon: 'fas fa-upload' },
  { name: '实时监听', route: '/monitor', icon: 'fas fa-signal' },
  { name: '用户管理', route: '/user', icon: 'fas fa-user-cog' }
]

function navigate(routePath) {
  router.push(routePath)
  if (window.innerWidth < 768) {
    collapsed.value = true
  }
}

function handleClickOutside(event) {
  if (!sidebarRef.value) return
  if (!sidebarRef.value.contains(event.target)) {
    if (window.innerWidth < 768) {
      collapsed.value = true
    }
  }
}

const handleResize = () => {
  if (window.innerWidth < 768) {
    collapsed.value = true
  }
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
  window.addEventListener('resize', handleResize)
  handleResize()
})

onBeforeUnmount(() => {
  document.removeEventListener('click', handleClickOutside)
  window.removeEventListener('resize', handleResize)
})
</script>

<style scoped>
.sidebar {
  position: fixed;
  top: 56px;
  left: 0;
  height: calc(100vh - 56px);
  width: 240px;
  background-color: #f8f9fa;
  border-right: 1px solid #e0e0e0;
  padding: 20px 12px;
  transition: width 0.3s ease;
  z-index: 1000;
  overflow-x: hidden;
}
.sidebar.collapsed {
  width: 80px;
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
  background-color: #ffffff;
  border-radius: 10px;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
  font-size: 15px;
  color: #444;
  cursor: pointer;
  transition: all 0.2s ease;
}
.nav-item:hover {
  background-color: #f0f4ff;
  color: #333;
}
.nav-item.active {
  background-color: #dbe9ff;
  color: #1a4db3;
  font-weight: 600;
}

.nav-item i {
  font-size: 16px;
  min-width: 20px;
  text-align: center;
}

.label {
  flex: 1;
  white-space: nowrap;
}

/* 小屏强制收起 */
@media (max-width: 768px) {
  .sidebar {
    width: 80px !important;
  }
}
</style>
