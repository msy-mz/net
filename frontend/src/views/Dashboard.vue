<template>
  <div class="dashboard-wrapper">
    <h2 class="main-title">系统总览</h2>

    <div class="grid">
      <div class="card-box" v-for="(item, index) in stats" :key="index">
        <div class="icon-wrap">
          <i :class="item.icon"></i>
        </div>
        <div class="info">
          <div class="label">{{ item.label }}</div>
          <div class="value">{{ item.value }}</div>
        </div>
      </div>
    </div>

    <div class="chart-card">
      <div class="chart-title chart-title-center">实时异常流量趋势图</div>
      <canvas id="realtimeAttackChart" height="100"></canvas>
    </div>


    <div class="grid-2 mt-4">
      <div class="chart-card">
        <div class="chart-title">近期攻击种类占比</div>
        <img src="http://localhost:8888/static/attack_type_bar.png" class="chart-img" />
      </div>

      <div class="chart-card">
        <div class="chart-title">Top5 攻击者 IP</div>
        <div class="ip-cards">
          <div class="ip-card" v-for="ip in topIps" :key="ip.ip">
            <div class="ip-main">
              <strong>{{ ip.ip }}</strong>
              <span class="badge">{{ ip.label }}</span>
            </div>
            <div class="ip-detail">次数：{{ ip.count }}，频率：{{ ip.rate }}/min</div>
            <div class="ip-detail">时间：{{ ip.time }}</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>

import { Chart, LineController, LineElement, PointElement, LinearScale, Title, CategoryScale } from 'chart.js'
import { onMounted, onUnmounted } from 'vue'

Chart.register(LineController, LineElement, PointElement, LinearScale, Title, CategoryScale)

let attackChart = null
let attackTimer = null
const attackData = Array(30).fill(0)

const initAttackChart = () => {
  const ctx = document.getElementById('realtimeAttackChart')?.getContext('2d')
  if (!ctx) return

  attackChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: Array(30).fill(''),
      datasets: [{
        label: '每秒攻击数量',
        data: attackData,
        borderColor: '#e74c3c',
        backgroundColor: 'rgba(231, 76, 60, 0.15)',
        pointRadius: 2,
        tension: 0.3,
        fill: true
      }]
    },
    options: {
      animation: false,
      responsive: true,
      plugins: {
        legend: {
          display: false
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: { stepSize: 1, color: '#444' }
        },
        x: {
          ticks: { color: '#444' }
        }
      }
    }
  })
}

const startAttackChart = () => {
  attackTimer = setInterval(() => {
    const newVal = Math.floor(Math.random() * 5) // 可替换为实际异常计数
    attackData.push(newVal)
    if (attackData.length > 30) attackData.shift()
    attackChart.update()
  }, 1000)
}

const stopAttackChart = () => {
  clearInterval(attackTimer)
}


import { ref } from 'vue'


const isListening = ref(false)

const toggleListening = () => {
  isListening.value = !isListening.value
  if (isListening.value) {
    startAttackChart()
  } else {
    stopAttackChart()
  }
}


const stats = ref([
  { label: '实时监听状态', value: '运行中', icon: 'fas fa-broadcast-tower' },
  { label: '今日推理流量', value: '1,284 条', icon: 'fas fa-network-wired' },
  { label: '异常检测数', value: '327 条', icon: 'fas fa-exclamation-triangle' },
  { label: '系统运行时长', value: '12h 45m', icon: 'fas fa-clock' }
])

const rings = [
  { label: 'CPU 使用率', value: 61, color: '#4c84ff' },
  { label: '内存使用率', value: 45, color: '#34c38f' },
  { label: '异常比率', value: 26.9, color: '#f46a6a' }
]

const topIps = ref([
  { ip: '192.168.1.103', count: 42, rate: 6, time: '2025-05-23 19:21', label: 'DoS Hulk' },
  { ip: '10.0.0.25', count: 36, rate: 5, time: '2025-05-23 19:18', label: 'PortScan' },
  { ip: '172.16.5.12', count: 31, rate: 4, time: '2025-05-23 18:59', label: 'DDoS' },
  { ip: '192.168.2.77', count: 28, rate: 4, time: '2025-05-23 18:45', label: 'SSH-Patator' },
  { ip: '10.1.1.55', count: 21, rate: 3, time: '2025-05-23 18:32', label: 'Web Attack' }
])

const getDash = (percent) => {
  const r = 40
  const c = 2 * Math.PI * r
  const filled = (percent / 100) * c
  return `${filled} ${c}`
}
</script>

<style scoped>
.dashboard-wrapper {
  max-width: 1180px;
  margin: 0 auto;
  font-family: 'Segoe UI', 'HarmonyOS Sans', 'Microsoft YaHei', sans-serif;
}
.module-title {
  font-size: 20px;
  font-weight: 700;
  color: #222;
  margin: 24px 0 16px 0;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 24px;
}
.grid-2 {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
  gap: 28px;
}
.card-box {
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.04);
  padding: 24px;
  display: flex;
  align-items: center;
  gap: 16px;
}
.icon-wrap {
  font-size: 24px;
  color: #4c84ff;
  background-color: #eaf0ff;
  border-radius: 50%;
  padding: 12px;
}
.info .label {
  font-size: 13px;
  color: #666;
}
.info .value {
  font-size: 18px;
  font-weight: 600;
  color: #222;
}
.chart-card {
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
  padding: 20px;
}

.chart-title-center {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 12px;
  text-align: center;
  color: #333;
}

.chart-title {
  font-size: 15px;
  font-weight: 600;
  margin-bottom: 12px;
  color: #333;
}
.chart-img {
  width: 100%;
  border-radius: 8px;
}
.circle-group {
  display: flex;
  gap: 24px;
  justify-content: space-around;
  margin-bottom: 10px;
}
.circle-item {
  text-align: center;
  position: relative;
}
.circle-text {
  margin-top: -82px;
  display: flex;
  flex-direction: column;
  align-items: center;
  pointer-events: none;
}
.circle-text .percent {
  font-size: 18px;
  font-weight: 600;
  color: #222;
}
.circle-text .label {
  font-size: 12px;
  color: #666;
}
.status-list {
  padding-left: 20px;
  font-size: 14px;
  color: #444;
  line-height: 1.8;
}
.ip-cards {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.ip-card {
  background: #f8f9fa;
  border-radius: 8px;
  padding: 12px 16px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.03);
}
.ip-main {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 14px;
  font-weight: 600;
}
.badge {
  background-color: #dfeaff;
  color: #4c84ff;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 12px;
}
.ip-detail {
  font-size: 13px;
  color: #555;
  margin-top: 2px;
}

.main-title {
  font-size: 22px;
  font-weight: 600;
  color: #222;
  margin-bottom: 6px;
  text-align: center;
}

</style>
