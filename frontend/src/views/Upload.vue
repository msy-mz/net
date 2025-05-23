<template>
  <div class="upload-wrapper">
    <h2 class="main-title">上传分析</h2>

    <div class="card-box">
      <label class="form-label">选择模型：</label>
      <select v-model="selectedModel" class="form-select mb-3">
        <option value="ft">FT-Encoder</option>
        <option value="id">ID-TCN</option>
        <option value="fusion">FT-IDNet</option>
      </select>

      <input type="file" multiple class="form-control mb-3" @change="handleFiles" />
      <button class="btn btn-primary w-100" :disabled="!files.length" @click="uploadPcap">上传文件</button>

      <div v-if="uploadResult.length" class="mt-4 text-start">
        <h6 class="mb-2">已上传路径：</h6>
        <ul class="upload-list">
          <li v-for="(path, index) in uploadResult" :key="index">{{ path }}</li>
        </ul>
      </div>
    </div>

    <div v-if="summary" class="result-area mt-4 text-start">
      <h6 class="mb-2">统计报表：</h6>
      <p>总流量数：{{ summary.total }}</p>
      <p>异常流量数：{{ summary.abnormal }}（{{ ((summary.abnormal / summary.total) * 100).toFixed(2) }}%）</p>
    </div>

    <div v-if="chartVisible" class="mt-4">
      <canvas id="barChart"></canvas>
    </div>

    <div v-if="detailedResults.length" class="table-responsive mt-4">
      <div class="d-flex justify-content-between align-items-center mb-2">
        <h6 class="mb-2 text-start">异常流元信息：</h6>
        <button class="btn btn-outline-secondary btn-sm" @click="downloadCSV">导出 CSV</button>
      </div>
      <table class="table table-striped table-hover table-sm">
        <thead>
          <tr>
            <th>时间</th>
            <th>源 IP</th>
            <th>目标 IP</th>
            <th>源端口</th>
            <th>目标端口</th>
            <th>类别</th>
            <th>置信度</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(item, index) in detailedResults" :key="index">
            <td>{{ item.timestamp }}</td>
            <td>{{ item.src_ip }}</td>
            <td>{{ item.dst_ip }}</td>
            <td>{{ item.src_port }}</td>
            <td>{{ item.dst_port }}</td>
            <td>{{ item.pred_label }}</td>
            <td>{{ (item.confidence * 100).toFixed(2) }}%</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>


<script>
import axios from 'axios'
import Chart from 'chart.js/auto'

export default {
  data() {
    return {
      files: [],
      uploadResult: [],
      selectedModel: 'ft',
      summary: null,
      detailedResults: [],
      chartVisible: false,
      chart: null
    }
  },
  methods: {
    handleFiles(event) {
      this.files = Array.from(event.target.files)
    },
    uploadPcap() {
      const formData = new FormData()
      this.files.forEach(file => formData.append('file', file))
      formData.append('model', this.selectedModel)

      axios.post('http://localhost:8888/upload/pcap', formData)
        .then(res => {
          this.uploadResult = res.data.pcap_paths
          this.summary = res.data.summary
          this.detailedResults = res.data.detailed_results
          this.drawChart(res.data.label_distribution)
        })
        .catch(() => {
          alert('上传失败')
        })
    },
    drawChart(labelDist) {
      this.chartVisible = true
      if (this.chart) this.chart.destroy()

      const ctx = document.getElementById('barChart')
      const labels = Object.keys(labelDist)
      const counts = Object.values(labelDist)

      this.chart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            label: '流量分类分布',
            data: counts,
            backgroundColor: '#007bff'
          }]
        },
        options: {
          responsive: true,
          plugins: {
            legend: { display: false }
          }
        }
      })
    },

    downloadCSV() {
    const headers = ['时间', '源IP', '目标IP', '源端口', '目标端口', '类别', '置信度']
    const rows = this.detailedResults.map(item => [
      item.timestamp,
      item.src_ip,
      item.dst_ip,
      item.src_port,
      item.dst_port,
      item.pred_label,
      (item.confidence * 100).toFixed(2) + '%'
    ])

    let csvContent = '\uFEFF' + headers.join(',') + '\n' + rows.map(e => e.join(',')).join('\n')
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })

    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.setAttribute('download', 'detailed_results.csv')
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }
  }
}
</script>


<style scoped>
.upload-wrapper {
  max-width: 820px;
  margin: 0 auto;
  text-align: center;
}
.main-title {
  font-size: 22px;
  font-weight: 600;
  color: #222;
  margin-bottom: 6px;
}
.card-box {
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.04);
  padding: 30px;
  text-align: left;
}
.upload-list {
  font-size: 14px;
  color: #555;
  padding-left: 16px;
}
.result-area {
  font-size: 15px;
}
.table th, .table td {
  vertical-align: middle;
  text-align: center;
}
</style>
