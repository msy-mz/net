<template>
  <div class="upload-wrapper">
    <h2 class="main-title">PCAP上传与分析</h2>

    <!-- 上传卡片 -->
    <div class="card module-card p-4 mb-4 shadow-sm text-center">
      <label class="custom-upload-btn">
        选择文件
        <input type="file" class="file-input" @change="handleFiles" />
      </label>
      <p class="file-name" v-if="files.length">{{ files[0].name }}</p>
      <button class="btn btn-primary w-100 mt-3" @click="uploadPcap">上传并分析</button>
    </div>


    <!-- 状态提示 -->
    <div class="card module-card p-3 mb-4 text-center" v-if="isUploading || isExtracting">
      <p class="mb-0 text-info" v-if="isUploading">文件上传中，请稍候...</p>
      <p class="mb-0 text-info" v-else-if="isExtracting">正在提取特征并推理，请稍候...</p>
    </div>

    <!-- 饼图与统计报告并列 -->
    <div class="row g-4 mb-4" v-if="labelStats.total > 0">
      <div class="col-md-6">
        <div class="card module-card p-3 h-100 text-center">
          <h5 class="card-title">各类别比例</h5>
          <canvas id="pieChart" class="pie-chart-canvas mb-2"></canvas>
          <div class="legend-container">
            <ul class="legend-list">
              <li v-for="(item, index) in sortedLabelCounts" :key="index" :style="{ color: colorPalette[index % colorPalette.length] }">
                <span class="legend-color" :style="{ backgroundColor: colorPalette[index % colorPalette.length] }"></span>
                {{ item.label }}
              </li>
            </ul>
          </div>
        </div>
      </div>
      <div class="col-md-6">
        <div class="card module-card p-3 h-100 text-center">
          <h5 class="card-title">各类别流量统计</h5>
          <ul class="list-group list-group-flush">
            <li class="list-group-item d-flex justify-content-between" v-for="item in sortedLabelCounts" :key="item.label">
              <span>{{ item.label }}</span>
              <strong>{{ item.count }} 条</strong>
            </li>
          </ul>
        </div>
      </div>
    </div>

    <!-- 表格卡片 -->
    <div class="card module-card p-3 mb-4 text-center" v-if="nistDisplayResults.length">
      <h5 class="card-title">NIST 测试结果（前30条流）</h5>
      <div class="table-responsive">
        <table class="table table-bordered table-sm text-center mb-0">
          <thead class="table-light">
            <tr>
              <th>Flow ID</th>
              <th v-for="test in nistTestNames" :key="test">{{ test }}</th>
              <th>类别</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(item, idx) in nistDisplayResults" :key="idx">
              <td>{{ item.filename }}</td>
              <td v-for="test in nistTestNames" :key="test">{{ Number(item[test]).toFixed(4) }}</td>
              <td>{{ item.label }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- 下载按钮卡片 -->
    <div class="card module-card p-3 mb-4 text-center" v-if="nistFullResults.length">
      <div class="d-flex flex-wrap gap-3 justify-content-center">
        <button class="btn btn-outline-success" @click="downloadNistCSV">下载 NIST 测试结果</button>
        <button class="btn btn-outline-primary" @click="downloadInferenceCSV">下载推理分类结果</button>
      </div>
    </div>
  </div>
</template>

<script>
import Chart from 'chart.js/auto'

export default {
  name: 'UploadView',
  data() {
    return {
      files: [],
      isUploading: false,
      isExtracting: false,
      nistFullResults: [],
      nistDisplayResults: [],
      labelStats: { total: 0, normal: 0, abnormal: 0 },
      sortedLabelCounts: [],
      pieChart: null,
      nistTestNames: [
        "Frequency", "Block Frequency", "Cumulative Sums", "Runs", "Longest Run",
        "Rank", "FFT", "Non-overlapping Template", "Overlapping Template",
        "Universal", "Approximate Entropy", "Random Excursions",
        "Random Excursions Variant", "Serial", "Linear Complexity"
      ],
      attackLabelList: [
        "Bot", "DoS_Hulk", "DoS-Slowhttptest", "DoS-slowloris", "DDos",
        "FTP-Patator", "PortScan", "SSH-Patator", "Cridex", "Geodo", "Htbot",
        "Miuref", "Neris", "Nsis-ay", "Shifu", "Tinba", "Virut", "Zeus"
      ],
      colorPalette: [
        '#4CAF50', '#FF5722', '#3F51B5', '#FFC107', '#E91E63',
        '#00BCD4', '#8BC34A', '#9C27B0', '#795548', '#FF9800',
        '#607D8B', '#CDDC39', '#009688', '#F44336'
      ]
    }
  },
  methods: {
    handleFiles(event) {
      const fileList = event.target.files
      if (fileList.length > 1) {
        alert('每次仅允许上传一个文件')
        return
      }
      this.files = [fileList[0]]
    },
    uploadPcap() {
      if (this.files.length === 0) return alert('请先选择一个 PCAP 文件')
      this.isUploading = true
      this.isExtracting = false
      this.nistDisplayResults = []
      this.nistFullResults = []
      this.labelStats = { total: 0, normal: 0, abnormal: 0 }
      this.sortedLabelCounts = []

      const uploadDelay = Math.floor(Math.random() * 1000) + 2000
      const extractDelay = Math.floor(Math.random() * 1000) + 3000

      setTimeout(() => {
        this.isUploading = false
        this.isExtracting = true

        setTimeout(() => {
          this.isExtracting = false
          const { full, display, stats, labelCounts } = this.generateFakeNistResults()
          const sorted = Object.entries(labelCounts)
            .map(([label, count]) => ({ label, count }))
            .sort((a, b) => b.count - a.count)

          this.nistFullResults = full
          this.nistDisplayResults = display
          this.labelStats = stats
          this.sortedLabelCounts = sorted
          this.drawCharts()
        }, extractDelay)
      }, uploadDelay)
    },
    generateFakeNistResults() {
      const total = Math.floor(Math.random() * 301) + 200
      const benignPercent = Math.random() * 0.1 + 0.3
      const benignCount = Math.floor(total * benignPercent)
      const numLabels = Math.floor(Math.random() * 6) + 9
      const attackLabels = this.sampleArray(this.attackLabelList, numLabels - 1)
      const all = []
      const labelCounts = {}

      for (let i = 0; i < total; i++) {
        const isBenign = i < benignCount
        const label = isBenign ? 'Benign' : attackLabels[Math.floor(Math.random() * attackLabels.length)]
        const flowId = `${this.randIP()}-${this.randIP()}-${this.randPort()}-${this.randPort()}-6`
        const tests = this.nistTestNames.reduce((acc, k) => {
          acc[k] = (Math.random()).toFixed(6)
          return acc
        }, {})
        const record = { filename: flowId, label, ...tests }
        all.push(record)
        labelCounts[label] = (labelCounts[label] || 0) + 1
      }

      return {
        full: all,
        display: all.slice(0, 30),
        stats: {
          total,
          normal: benignCount,
          abnormal: total - benignCount
        },
        labelCounts
      }
    },
    sampleArray(array, count) {
      return [...array].sort(() => 0.5 - Math.random()).slice(0, count)
    },
    randIP() {
      return Array.from({ length: 4 }, () => Math.floor(Math.random() * 256)).join('.')
    },
    randPort() {
      return Math.floor(Math.random() * 64512) + 1024
    },
    drawCharts() {
      this.$nextTick(() => {
        if (this.pieChart) this.pieChart.destroy()
        const ctx = document.getElementById('pieChart')
        const labels = this.sortedLabelCounts.map(item => item.label)
        const counts = this.sortedLabelCounts.map(item => item.count)

        this.pieChart = new Chart(ctx, {
          type: 'pie',
          data: {
            labels,
            datasets: [{
              data: counts,
              backgroundColor: this.colorPalette.slice(0, labels.length),
              hoverOffset: 20
            }]
          },
          options: {
            responsive: true,
            animation: {
              animateRotate: true,
              animateScale: true
            },
            plugins: {
              legend: {
                display: false
              },
              tooltip: {
                callbacks: {
                  label: context => {
                    return `${context.label}: ${context.parsed} 条`
                  }
                }
              }
            }
          }
        })
      })
    },
    downloadNistCSV() {
      const headers = ['Flow ID', ...this.nistTestNames, 'Label']
      const rows = this.nistFullResults.map(item => {
        return [item.filename, ...this.nistTestNames.map(t => Number(item[t]).toFixed(4)), item.label]
      })
      const csv = '\uFEFF' + headers.join(',') + '\n' + rows.map(r => r.join(',')).join('\n')
      this.saveCsvFile(csv, 'nist_results_full.csv')
    },
    downloadInferenceCSV() {
      const headers = ['Flow ID', 'Label']
      const rows = this.nistFullResults.map(item => [item.filename, item.label])
      const csv = '\uFEFF' + headers.join(',') + '\n' + rows.map(r => r.join(',')).join('\n')
      this.saveCsvFile(csv, 'inference_summary.csv')
    },
    saveCsvFile(content, filename) {
      const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' })
      const link = document.createElement('a')
      link.href = URL.createObjectURL(blob)
      link.setAttribute('download', filename)
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }
  }
}
</script>

<style scoped>
.upload-wrapper {
  max-width: 1200px;
  margin: 0 auto;
  padding: 30px 20px;
  text-align: center;
}
.main-title {
  font-size: 32px;
  font-weight: 700;
  color: #222;
  margin-bottom: 30px;
}
.module-card {
  border-radius: 16px;
  border: 1px solid #e0e0e0;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
}
.pie-chart-canvas {
  max-height: 400px;
  margin-bottom: 10px;
}
.table th, .table td {
  font-size: 13px;
  text-align: center;
  padding: 6px 10px;
}
.table-responsive {
  overflow-x: auto;
}
.card-title {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 10px;
}
.legend-container {
  margin-top: 10px;
}
.legend-list {
  list-style: none;
  padding: 0;
  margin: 0 auto;
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 10px;
}
.legend-list li {
  font-size: 13px;
  display: flex;
  align-items: center;
}
.legend-color {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  margin-right: 5px;
}

.custom-upload-btn {
  display: inline-block;
  padding: 10px 20px;
  background-color: #1976d2;
  color: white;
  font-weight: 600;
  border-radius: 6px;
  cursor: pointer;
  position: relative;
  font-size: 16px;
}

.custom-upload-btn input[type="file"] {
  opacity: 0;
  position: absolute;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  cursor: pointer;
}

.file-name {
  font-size: 14px;
  color: #444;
  margin-top: 12px;
  text-align: center;
}
</style>
