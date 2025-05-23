<template>
  <div class="log-wrapper">
    <h2 class="main-title">监听日志</h2>
    <!-- <p class="desc">系统监听模块捕获的实时预测记录</p> -->

    <div class="log-card">
      <div v-if="logs.length" class="table-responsive">
        <table class="table table-hover table-sm">
          <thead>
            <tr>
              <th>时间戳</th>
              <th>源 IP</th>
              <th>目标 IP</th>
              <th>预测类别</th>
              <th>置信度</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(item, index) in logs" :key="index">
              <td>{{ item.timestamp }}</td>
              <td>{{ item.src_ip }}</td>
              <td>{{ item.dst_ip }}</td>
              <td>{{ item.pred_label }}</td>
              <td>{{ (item.confidence * 100).toFixed(2) }}%</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div v-else class="no-log">暂无日志记录</div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const logs = ref([])

onMounted(() => {
  axios.get('http://localhost:8888/result/log.json')
    .then(res => {
      logs.value = res.data
    })
    .catch(() => {})
})
</script>

<style scoped>
.log-wrapper {
  max-width: 1080px;
  margin: 0 auto;
  text-align: center;
}
.main-title {
  font-size: 22px;
  font-weight: 600;
  color: #222;
  margin-bottom: 6px;
}
.desc {
  font-size: 14px;
  color: #888;
  margin-bottom: 24px;
}
.log-card {
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.04);
  padding: 30px;
  text-align: left;
}
.no-log {
  font-size: 14px;
  color: #999;
  text-align: center;
}
.table th {
  color: #555;
  font-weight: 500;
}
</style>
