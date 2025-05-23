<template>
  <div class="nist-container">
    <div class="card">
      <h5 class="card-title">NIST 随机性测试</h5>
      <!-- <p class="card-desc">上传一个二进制文件，运行 NIST SP800-22 随机性测试。</p> -->

      <input type="file" class="form-control" @change="handleFile" />
      <button class="btn btn-primary mt-3" :disabled="!file" @click="submitFile">提交测试</button>

      <div v-if="loading" class="loading-text">正在检测，请稍候...</div>

      <div v-if="results && Object.keys(results).length" class="result-card">
        <h6>测试结果：</h6>
        <table class="table table-hover table-sm">
          <thead>
            <tr>
              <th>测试项</th>
              <th>P 值</th>
              <th>结果</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(value, key) in results" :key="key">
              <td>{{ key }}</td>
              <td>{{ value.toFixed(3) }}</td>
              <td>
                <span :class="value >= 0.01 ? 'pass' : 'fail'">
                  {{ value >= 0.01 ? '通过' : '未通过' }}
                </span>
              </td>
            </tr>
          </tbody>
        </table>
        <button class="btn btn-outline-secondary mt-2" @click="downloadCSV">导出结果 CSV</button>
      </div>

      <div v-if="error" class="text-danger mt-2">{{ error }}</div>
    </div>
  </div>
</template>

<script setup>
import request from '@/utils/request'
import { ref } from 'vue'

const file = ref(null)
const results = ref({})
const error = ref('')
const loading = ref(false)

const handleFile = (e) => {
  file.value = e.target.files[0]
}

const submitFile = () => {
  if (!file.value) return
  const formData = new FormData()
  formData.append('file', file.value)
  results.value = {}
  error.value = ''
  loading.value = true

  request.post('/api/nist/test', formData)
    .then(res => {
      results.value = res.data.result
    })
    .catch(err => {
      error.value = err.response?.data?.error || '检测失败'
    })
    .finally(() => {
      loading.value = false
    })
}

const downloadCSV = () => {
  const headers = ['测试项', 'P 值']
  const rows = Object.entries(results.value)
  let csvContent = headers.join(',') + '\n'
  rows.forEach(([name, value]) => {
    csvContent += `${name},${value}\n`
  })
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.setAttribute('href', url)
  link.setAttribute('download', `nist_result_${file.value.name}.csv`)
  link.click()
  URL.revokeObjectURL(url)
}
</script>

<style scoped>
.nist-container {
  max-width: 800px;
  margin: 0 auto;
}
.card {
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.04);
  padding: 30px;
}
.card-title {
  font-size: 18px;
  font-weight: 600;
  color: #333;
  margin-bottom: 10px;
}
.card-desc {
  font-size: 14px;
  color: #666;
  margin-bottom: 20px;
}
.result-card {
  margin-top: 30px;
}
.table th {
  font-weight: 500;
  color: #666;
}
.pass {
  color: #2e7d32;
  font-weight: 500;
}
.fail {
  color: #d32f2f;
  font-weight: 500;
}
.loading-text {
  margin-top: 16px;
  color: #888;
  font-size: 14px;
}
</style>
