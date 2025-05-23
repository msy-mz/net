<template>
  <div class="row">
    <div class="col-md-4" v-for="(card, index) in cards" :key="index">
      <div class="card shadow-sm">
        <div class="card-body">
          <h6 class="text-muted">
            <i :class="card.icon" class="me-2"></i>{{ card.title }}
          </h6>
          <component :is="card.component" v-bind="card.data" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
// 系统统计卡片配置，包含图标类名
const cards = [
  {
    title: '今日流量分析',
    icon: 'fas fa-chart-area',
    component: {
      template: `<div>
        <p class="display-6 text-primary fw-bold">12,431</p>
        <div class="text-success">↑ +8.4% 较昨日</div>
      </div>`
    },
    data: {}
  },
  {
    title: '当前运行模型',
    icon: 'fas fa-microchip',
    component: {
      template: `<div>
        <p class="fs-5">FT-IDNet++</p>
        <span class="badge bg-success">运行中</span>
      </div>`
    },
    data: {}
  },
  {
    title: '系统资源使用',
    icon: 'fas fa-memory',
    component: {
      template: `<div>
        <div class="progress mb-2">
          <div class="progress-bar" role="progressbar" style="width: 60%;">CPU 60%</div>
        </div>
        <div class="progress">
          <div class="progress-bar bg-warning" role="progressbar" style="width: 40%;">内存 40%</div>
        </div>
      </div>`
    },
    data: {}
  }
]
</script>
