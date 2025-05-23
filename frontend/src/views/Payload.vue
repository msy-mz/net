<template>
  <div class="payload-wrapper">
    <h2 class="main-title">载荷提取</h2>
    <p class="desc">支持从上传的 PCAP 文件中提取 TCP/UDP Payload</p>

    <div class="card-box">
      <input type="file" class="form-control mb-3" @change="handleFile" accept=".pcap,.pcapng" />
      <button class="btn btn-primary w-100" @click="uploadPcap" :disabled="!file">开始提取</button>

      <div v-if="payloads.length" class="mt-4">
        <h6 class="mb-2">提取结果（部分展示）：</h6>
        <table class="table table-sm table-bordered">
          <thead>
            <tr><th>#</th><th>协议</th><th>源IP</th><th>目标IP</th><th>Payload（前64字节）</th></tr>
          </thead>
          <tbody>
            <tr v-for="(item, index) in payloads.slice(0, 10)" :key="index">
              <td>{{ index + 1 }}</td>
              <td>{{ item.protocol }}</td>
              <td>{{ item.src_ip }}</td>
              <td>{{ item.dst_ip }}</td>
              <td><code>{{ item.payload_hex }}</code></td>
            </tr>
          </tbody>
        </table>
        <button class="btn btn-outline-secondary btn-sm" @click="download">导出全部结果</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'
import { saveAs } from 'file-saver'

const file = ref(null)
const payloads = ref([])

function handleFile(e) {
  file.value = e.target.files[0]
}

async function uploadPcap() {
  const formData = new FormData()
  formData.append('file', file.value)

  const res = await axios.post('/api/payload/extract', formData)
  payloads.value = res.data.payloads
}

function download() {
  const blob = new Blob([JSON.stringify(payloads.value, null, 2)], { type: 'application/json' })
  saveAs(blob, 'payload_extract.json')
}
</script>

<style scoped>
.payload-wrapper {
  max-width: 960px;
  margin: 0 auto;
  text-align: center;
}
.main-title {
  font-size: 22px;
  font-weight: 600;
  margin-bottom: 6px;
}
.desc {
  font-size: 14px;
  color: #888;
  margin-bottom: 24px;
}
.card-box {
  background: #fff;
  border-radius: 12px;
  padding: 30px;
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.04);
  text-align: left;
}
</style>
