<template>
  <div class="card shadow-sm p-3">
    <h6 class="mb-3">
      <i class="fas fa-wave-square me-2"></i>今日异常流量趋势图
    </h6>
    <canvas ref="trendChart" height="120" aria-label="异常流量趋势图" role="img"></canvas>
  </div>
</template>

<script setup>
// 引入 Chart.js 及注册组件
import { onMounted, ref } from 'vue'
import { Chart, LineElement, LineController, CategoryScale, LinearScale, PointElement, Title, Tooltip } from 'chart.js'

Chart.register(LineElement, LineController, CategoryScale, LinearScale, PointElement, Title, Tooltip)

const trendChart = ref(null)

// 模拟数据：每 2 小时流量波动值
const labels = ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00']
const data = [5, 9, 7, 6, 4, 8, 12, 15, 10, 6, 8, 9]

onMounted(() => {
  new Chart(trendChart.value, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: '异常流量数量',
        data: data,
        borderColor: 'rgba(75, 192, 192, 1)',
        tension: 0.3,
        fill: false,
        pointRadius: 4
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { mode: 'index', intersect: false }
      },
      scales: {
        x: { display: true, title: { display: true, text: '时间段' } },
        y: { display: true, title: { display: true, text: '流量数量' } }
      }
    }
  })
})
</script>
