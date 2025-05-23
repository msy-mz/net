<template>
  <aside :class="['sidebar', { collapsed }]">
    <nav v-show="!collapsed || isDesktop">
      <ul>
        <li v-for="item in navs" :key="item.name"
            :class="{ active: $route.path === item.route }"
            @click="navigate(item.route)">
          <i :class="item.icon"></i>
          <span>{{ item.name }}</span>
        </li>
      </ul>
    </nav>
  </aside>
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
  { name: '系统总览', route: '/', icon: 'fas fa-chart-pie' },
  { name: '实时监听', route: '/monitor', icon: 'fas fa-broadcast-tower' },
  { name: '上传分析', route: '/upload', icon: 'fas fa-file-upload' },
  { name: '可视化分析', route: '/visual', icon: 'fas fa-project-diagram' },
  { name: '监听日志', route: '/log', icon: 'fas fa-file-alt' },
  { name: '载荷提取', route: '/payload', icon: 'fas fa-database' },
  { name: 'NIST 测试', route: '/nist-test', icon: 'fas fa-microscope' },
  { name: '用户管理', route: '/user', icon: 'fas fa-users-cog' }// 新增

]


const navigate = (route) => {
  if (!isDesktop.value) collapsed.value = true
  router.push(route)
}
</script>

<style scoped>
.sidebar {
  width: 240px;
  background-color: #ffffff;
  color: #333;
  height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: 0 16px;
  border-right: 1px solid #e6e6e6;
  font-family: "Segoe UI", "HarmonyOS Sans", sans-serif;
}
nav ul {
  list-style: none;
  padding: 0;
  margin: 0 auto;
  width: 100%;
}
nav ul li {
  display: flex;
  align-items: center;
  padding: 16px 20px;
  font-size: 16px;
  cursor: pointer;
  border-radius: 10px;
  margin-bottom: 12px;
  transition: background-color 0.2s ease;
}
nav ul li i {
  margin-right: 12px;
  font-size: 16px;
  color: #4c84ff;
}
nav ul li:hover {
  background-color: #f2f4f8;
}
nav ul li.active {
  background-color: #eaf0ff;
  font-weight: 600;
}
nav ul li span {
  flex: 1;
}
</style>
