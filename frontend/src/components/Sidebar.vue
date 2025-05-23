<template>
  <aside ref="sidebarRef" :class="['sidebar', { collapsed }]">
    <nav>
      <ul>
        <li
          v-for="item in navs"
          :key="item.name"
          :class="{ active: $route.path === item.route }"
          @click="navigate(item.route)"
        >
          <i :class="item.icon"></i>
          <span v-show="!collapsed" class="label">{{ item.name }}</span>
        </li>
      </ul>
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
  { name: '上传分析', route: '/upload', icon: 'fas fa-file-upload' },
  { name: '实时监听', route: '/monitor', icon: 'fas fa-broadcast-tower' },
  { name: '用户管理', route: '/user', icon: 'fas fa-users-cog' }
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
  top: 56px; /* 与顶部栏对齐 */
  left: 0;
  height: calc(100vh - 56px);
  width: 220px;
  background-color: #1e1e2f;
  color: #fff;
  padding: 20px 12px;
  border-top-right-radius: 14px;
  border-bottom-right-radius: 14px;
  box-shadow: 4px 0 20px rgba(0, 0, 0, 0.15);
  transition: width 0.3s ease;
  z-index: 1000;
  overflow: hidden;
}

.sidebar.collapsed {
  width: 72px;
}

nav ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

nav ul li {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px;
  font-size: 15px;
  color: #ccc;
  cursor: pointer;
  border-radius: 10px;
  margin-bottom: 10px;
  transition: background-color 0.2s ease, color 0.2s ease;
}

nav ul li:hover {
  background-color: #2a2a40;
  color: #fff;
}

nav ul li.active {
  background-color: #3b4b73;
  color: #fff;
  font-weight: 600;
}

nav ul li i {
  font-size: 16px;
  min-width: 20px;
  text-align: center;
}

.label {
  flex: 1;
  white-space: nowrap;
}

/* 小屏自动收起 */
@media (max-width: 768px) {
  .sidebar {
    width: 72px !important;
  }
}
</style>
