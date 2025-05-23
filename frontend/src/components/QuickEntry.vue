<template>
  <div class="row g-3 mb-4">
    <div
      v-for="item in modules"
      :key="item.module"
      class="col-md-3"
    >
      <div
        class="card quick-entry shadow-sm border-0 text-white"
        :class="item.bgClass"
        @click="navigate(item.module)"
      >
        <div class="card-body">
          <h5 class="card-title">
            <i :class="item.icon" class="me-2"></i>{{ item.title }}
          </h5>
          <p class="card-text">{{ item.desc }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
// 快捷入口配置项，含图标类名
const modules = [
  { module: 'monitor', title: '实时监听', desc: '实时抓包与识别', bgClass: 'bg-primary', icon: 'fas fa-wifi' },
  { module: 'upload', title: '上传分析', desc: '上传 .pcap 文件', bgClass: 'bg-success', icon: 'fas fa-upload' },
  { module: 'visual', title: '可视化分析', desc: '查看降维聚类图', bgClass: 'bg-info', icon: 'fas fa-chart-line' },
  { module: 'log', title: '监听日志', desc: '查看预测记录', bgClass: 'bg-secondary', icon: 'fas fa-clipboard-list' }
]

// 导航方法，可用于跳转或触发事件
const navigate = (module) => {
  console.log('跳转模块:', module)
  // 示例跳转：router.push({ name: module }) 或 emit('navigate', module)
}
</script>
