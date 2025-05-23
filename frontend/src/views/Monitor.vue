<template>
  <div class="monitor-wrapper">
    <h2 class="main-title">实时监听</h2>

    <!-- 控制区域 -->
    <div class="card-area">
      <div class="status-line">
        <span>监听状态：</span>
        <strong :class="store.isRunning ? 'active' : (isStarting ? 'starting' : 'stopped')">
          {{ isStarting ? '正在启动，请稍候...' : (store.isRunning ? '监听中...' : '已停止') }}
        </strong>
      </div>

      <div class="btn-group mt-3">
        <button class="btn btn-success" @click="startListening" :disabled="store.isRunning || isStarting">启动监听</button>
        <button class="btn btn-outline-secondary" @click="stopListening" :disabled="!store.isRunning">停止监听</button>
        <button class="btn btn-outline-primary" @click="downloadCSV" :disabled="!store.dataList.length">下载日志</button>
      </div>
    </div>

    <!-- 摘要统计 -->
    <div class="summary-area">
      <p><strong>近期攻击流：</strong> {{ attackCount }} / {{ totalCount }} （异常占比 {{ attackRate }}%）</p>
    </div>

    <!-- 表格展示 -->
    <div class="table-container">
      <table class="table table-striped table-hover">
        <thead>
          <tr>
            <th>时间</th>
            <th>源IP</th>
            <th>目的IP</th>
            <th>源端口</th>
            <th>目的端口</th>
            <th>类别</th>
            <th>置信度</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(item, index) in displayedData" :key="index">
            <td>{{ item.time }}</td>
            <td>{{ item.src_ip }}</td>
            <td>{{ item.dst_ip }}</td>
            <td>{{ item.src_port }}</td>
            <td>{{ item.dst_port }}</td>
            <td>{{ item.label }}</td>
            <td>{{ (item.confidence * 100).toFixed(2) }}%</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- 异常趋势图 -->
    <div class="chart-area mt-4">
      <h5 class="chart-title">异常流量趋势图</h5>
      <canvas id="attackLineChart" height="100"></canvas>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount, onActivated, onDeactivated, nextTick } from 'vue'
import { useListenStore } from '@/stores/listenStore'
import { saveAs } from 'file-saver'
import { Chart, LineController, LineElement, PointElement, LinearScale, Title, CategoryScale } from 'chart.js'

Chart.register(LineController, LineElement, PointElement, LinearScale, Title, CategoryScale)

const store = useListenStore()
const isStarting = ref(false)
let chart = null
let chartTimer = null

const initChart = () => {
  nextTick(() => {
    const ctx = document.getElementById('attackLineChart')?.getContext('2d')
    if (!ctx) return

    chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],  // 初始化为空
        datasets: [{
          label: '每秒攻击数量',
          data: [],
          borderWidth: 2,
          borderColor: '#e53935',
          backgroundColor: 'rgba(229, 57, 53, 0.15)',
          pointRadius: 3,
          pointBackgroundColor: '#e53935',
          pointHoverRadius: 5,
          tension: 0.3,
          fill: true
        }]
      },
      options: {
        animation: false,
        responsive: true,
        plugins: {
          legend: {
            display: true,
            labels: { color: '#444', font: { size: 13 } }
          }
        },
        scales: {
          x: {
            ticks: {
              color: '#666',
              font: { size: 12 },
              maxRotation: 60,
              autoSkip: true
            }
          },
          y: {
            beginAtZero: true,
            ticks: {
              color: '#666',
              stepSize: 1,
              font: { size: 12 }
            }
          }
        }
      }
    })
  })
}



const updateChartLoop = () => {
  if (!chart) return
  chartTimer = setInterval(() => {
    // 确保按时间升序排序，再取最后30条
    const sorted = [...store.dataList].sort((a, b) =>
      new Date(a.time) - new Date(b.time)
    )
    const recentItems = sorted.slice(-30)

    const labels = recentItems.map(item =>
      item.time ? item.time.slice(11) : '--:--:--'
    )
    const values = recentItems.map(item =>
      item.label !== 'Benign' ? 1 : 0
    )

    chart.data.labels = labels
    chart.data.datasets[0].data = values
    chart.update()
  }, 1000)
}



const startListening = () => {
  isStarting.value = true
  setTimeout(() => {
    store.start()
    initChart()
    updateChartLoop()
    isStarting.value = false
  }, 4000 + Math.floor(Math.random() * 1000))
}

const stopListening = () => {
  store.stop()
  clearInterval(chartTimer)
  chartTimer = null
}

onActivated(() => {
  if (store.isRunning && !chartTimer) {
    updateChartLoop()
  }
})

onDeactivated(() => {
  clearInterval(chartTimer)
  chartTimer = null
})

// 显示真实时间
const safeParseTime = (timestamp, fallback = '无效时间') => {
  try {
    const real = new Date(timestamp)
    if (isNaN(real.getTime())) return fallback
    return real.toISOString().replace('T', ' ').slice(0, 19)
  } catch {
    return fallback
  }
}

const totalCount = computed(() => store.dataList.length)
const attackCount = computed(() => store.dataList.filter(d => d.label !== 'Benign').length)
const attackRate = computed(() => {
  return totalCount.value === 0 ? 0 : ((attackCount.value / totalCount.value) * 100).toFixed(1)
})

const displayedData = computed(() => {
  return [...store.dataList]
    .map(item => ({ ...item, time: safeParseTime(item.time) }))
    .sort((a, b) => {
      if (a.time === '无效时间') return 1
      if (b.time === '无效时间') return -1
      return new Date(b.time) - new Date(a.time)
    })
})

const downloadCSV = () => {
  const headers = ['时间', '源IP', '目的IP', '源端口', '目的端口', '类别', '置信度']
  
  const rows = store.dataList.map(item => [
    safeParseTime(item.time),
    item.src_ip,
    item.dst_ip,
    String(item.src_port),          // 转为字符串，防止科学计数
    String(item.dst_port),
    item.label,
    (item.confidence * 100).toFixed(2) + '%'
  ])

  const infoRow = ['监听IP: 10.30.247.240', '监听端口: 443']
  
  const csvContent = '\uFEFF' + [infoRow, headers, ...rows]
    .map(row => row.map(col => `"${col}"`).join(','))  // 加双引号确保安全
    .join('\n')

  const timestamp = new Date().toISOString().replace(/[:T]/g, '-').split('.')[0]
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
  saveAs(blob, `realtime_log_${timestamp}.csv`)
}


onMounted(() => {
  initChart()
  if (store.isRunning) updateChartLoop()
})

onBeforeUnmount(() => {
  clearInterval(chartTimer)
  chartTimer = null
})
</script>

<style scoped>
.monitor-wrapper {
  max-width: none;
  width: 100%;
  padding: 24px 48px;
  box-sizing: border-box;
  text-align: center;
  background-color: #fafafa;
}

.main-title {
  font-size: 26px;
  font-weight: 600;
  color: #222;
  margin-bottom: 20px;
}

.card-area,
.summary-area,
.table-container,
.chart-area {
  max-width: 1400px;
  margin: 0 auto 32px auto;
}

.card-area {
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.04);
  padding: 32px;
}

.status-line {
  font-size: 16px;
  margin-bottom: 20px;
  color: #555;
}

.status-line .active {
  color: #2e7d32;
}

.status-line .stopped {
  color: #d32f2f;
}

.status-line .starting {
  color: #f57c00;
}

.btn-group {
  display: flex;
  justify-content: center;
  gap: 24px;
  margin-top: 12px;
}

.summary-area {
  font-size: 16px;
  color: #444;
  padding-top: 8px;
}

.table-container {
  max-height: 420px;
  overflow-y: auto;
  overflow-x: auto;
  margin-top: 20px;
  border: 1px solid #eee;
  border-radius: 10px;
  background: #fff;
  padding: 12px;
}

table {
  width: 100%;
  table-layout: fixed;
}

th, td {
  font-size: 14px;
  text-align: center;
  white-space: nowrap;
  padding: 10px 4px;
}

.chart-title {
  text-align: center;
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 12px;
  color: #333;
}

.chart-area {
  background: #fff;
  border-radius: 12px;
  padding: 28px;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
}
</style>
