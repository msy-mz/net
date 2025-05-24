
<template>
  <div class="spectrogram-wrapper text-center">
    <h2 class="main-title">频谱分析</h2>

    <!-- 上传按钮卡片 -->
    <div class="upload-card">
      <div class="btn-row">
        <label class="btn-wide select-btn">
          选择 PCAP 文件
          <input type="file" accept=".pcap" @change="handleFile" hidden />
        </label>
      </div>
      <div v-if="filename" class="filename-display">{{ filename }}</div>
      <div class="btn-row mt-2">
        <button class="btn-wide analyze-btn" @click="analyzeFile" :disabled="!selectedFile || isUploading || isAnalyzing">
          {{ isUploading ? '文件上传中...' : (isAnalyzing ? '正在分析...' : '上传并分析') }}
        </button>
      </div>
    </div>

    <!-- 输出图卡片 -->
    <div class="grid-gallery">
      <div v-for="(item, index) in spectrograms" :key="index" class="chart-box">
        <canvas
          :ref="el => drawSpectrogram(el, item.data, index)"
          :id="'canvas-' + index"
          width="240"
          height="80"
          @click="openPreview(index)"
          class="previewable"
        ></canvas>
        <div class="label">{{ item.label }}</div>
        <div class="summary">均值：{{ item.stats.mean.toFixed(4) }}｜熵：{{ item.stats.entropy.toFixed(4) }}</div>
      </div>
    </div>

    <!-- 图表与导出 -->
    <div class="text-center mt-4 mb-4" v-if="spectrograms.length">
      <button class="btn btn-outline-primary export-btn" @click="exportAll">导出所有频谱图 PNG</button>
      <button class="btn btn-outline-secondary export-btn ml-2" @click="exportCSV">导出统计数据 CSV</button>
    </div>


    <!-- 图像放大弹窗 -->
    <div v-if="previewIndex !== null" class="overlay" @click.self="closePreview">
      <div class="preview-dialog">
        <h5>{{ spectrograms[previewIndex].label }} 频谱图</h5>
        <canvas
          :ref="el => drawSpectrogram(el, spectrograms[previewIndex].data, 'preview')"
          id="preview-canvas"
          width="720"
          height="240"
        ></canvas>
        <button class="btn btn-secondary mt-3" @click="closePreview">关闭预览</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import JSZip from 'jszip'
import { saveAs } from 'file-saver'

// === 类别定义 ===
const CLASS_LIST = [
  'Benign', 'Bot', 'DoS_Hulk', 'DoS-Slowhttptest', 'DoS-slowloris', 'DDoS',
  'FTP-Patator', 'PortScan', 'SSH-Patator', 'Cridex', 'Geodo', 'Htbot',
  'Shifu', 'Tinba', 'Virut', 'Zeus'
]

// === 状态变量 ===
const spectrograms = ref([])
const filename = ref('')
const selectedFile = ref(null)
const isUploading = ref(false)
const isAnalyzing = ref(false)
const previewIndex = ref(null)

// === 处理文件选择 ===
function handleFile(event) {
  const file = event.target.files[0]
  if (file) {
    filename.value = file.name
    selectedFile.value = file
  }
}

function exportCSV() {
  let csv = 'Label,Mean,Entropy\n'
  for (const item of spectrograms.value) {
    csv += `${item.label},${item.stats.mean.toFixed(6)},${item.stats.entropy.toFixed(6)}\n`
  }

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
  const link = document.createElement('a')
  link.href = URL.createObjectURL(blob)
  link.download = 'spectrogram_stats.csv'
  link.click()
}


// === 触发分析模拟流程 ===
async function analyzeFile() {
  isUploading.value = true
  await delay(randomInt(2000, 3000))
  isUploading.value = false

  isAnalyzing.value = true
  await delay(randomInt(1000, 3000))
  isAnalyzing.value = false

  generateSpectrograms()


}

function generateSpectrograms() {
  spectrograms.value = CLASS_LIST.map(label => {
    const data = generateMatrix(label)
    const mean = data.flat().reduce((a, b) => a + b, 0) / (12 * 32)
    const entropy = computeEntropy(data)
    return { label, data, stats: { mean, entropy } }
  })
  updateCharts()
}


// === 延迟模拟工具 ===
function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}
function randomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min
}

// === 模拟频谱图数据 ===
function generateMatrix(label) {
  const rows = 12, cols = 32
  const base = Array.from({ length: rows }, () => new Array(cols).fill(0))

  if (label === 'Benign') {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        base[r][c] = Math.exp(-c / 8) * 0.3 + Math.random() * 0.01
      }
    }

  } else if (label === 'DoS_Hulk') {
    for (let r = 0; r < rows; r++) {
      const amp = 0.4 + r * 0.03
      for (let c = 0; c < cols; c++) {
        base[r][c] = Math.abs(Math.sin(c * 0.8 + r * 0.3)) * amp + Math.random() * 0.01
      }
    }

  } else if (label === 'DoS-Slowhttptest' || label === 'DoS-slowloris') {
    for (let r = 0; r < rows; r++) {
      const phase = r * 0.5
      for (let c = 0; c < cols; c++) {
        base[r][c] = Math.abs(Math.sin((c + phase) * 0.3)) * 0.35 + Math.random() * 0.015
      }
    }

  } else if (label === 'DDoS') {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        base[r][c] = (
          Math.sin(c * 0.4 + r * 0.1) * 0.2 +
          Math.sin(c * 0.9 + r * 0.3) * 0.1 +
          Math.sin(c * 1.5 + r * 0.05) * 0.1
        ) + Math.random() * 0.015
      }
    }

  } else if (label === 'PortScan') {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (c > 24) {
          base[r][c] = 0.55 + 0.1 * Math.sin(r * 0.5 + c * 0.3) + Math.random() * 0.01
        } else {
          base[r][c] = 0.1 + 0.05 * Math.sin(c * 0.4 + r) + Math.random() * 0.01
        }
      }
    }

  } else if (label === 'FTP-Patator' || label === 'SSH-Patator') {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        base[r][c] = 0.15 + Math.abs(Math.sin(c * 0.6 + r * 0.3)) * 0.2 + Math.random() * 0.01
      }
    }

  } else if (['Bot', 'Htbot', 'Geodo', 'Cridex'].includes(label)) {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const w1 = Math.sin(c * 0.5 + r * 0.2) * 0.25
        const w2 = Math.cos(c * 0.3 + r) * 0.15
        base[r][c] = 0.1 + Math.abs(w1 + w2) + Math.random() * 0.015
      }
    }

  } else if (['Shifu', 'Tinba', 'Virut', 'Zeus'].includes(label)) {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        base[r][c] = 0.18 +
          Math.sin(c * 0.4 + r * 0.2) * 0.15 +
          Math.sin(r * 0.3 + c * 0.1) * 0.05 +
          Math.random() * 0.015
      }
    }

  } else {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        base[r][c] = 0.2 +
          Math.sin(c * 0.6 + r * 0.5) * 0.2 +
          Math.random() * 0.01
      }
    }
  }

  return base
}



// === 熵计算 ===
function computeEntropy(arr) {
  const flat = arr.flat()
  const bins = new Array(20).fill(0)
  flat.forEach(v => {
    const idx = Math.min(19, Math.floor(v * 20))
    bins[idx]++
  })
  const prob = bins.map(c => c / flat.length)
  return -prob.reduce((sum, p) => (p > 0 ? sum + p * Math.log2(p) : sum), 0)
}

// === 频谱图渲染 ===
function drawSpectrogram(canvas, matrix, index) {
  if (!canvas || !matrix) return
  const ctx = canvas.getContext('2d')
  const rows = matrix.length
  const cols = matrix[0].length
  const cellW = canvas.width / cols
  const cellH = canvas.height / rows
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const value = matrix[r][c]
      ctx.fillStyle = getColor(value)
      ctx.fillRect(c * cellW, r * cellH, cellW, cellH)
    }
  }
}

function getColor(val) {
  const h = (1 - val) * 280
  return `hsl(${h}, 100%, 50%)`
}

// === 导出为 zip ===
async function exportAll() {
  const zip = new JSZip()
  for (let index = 0; index < spectrograms.value.length; index++) {
    const canvas = document.getElementById('canvas-' + index)
    if (!canvas) continue
    const dataUrl = canvas.toDataURL('image/png')
    const base64 = dataUrl.split(',')[1]
    const label = spectrograms.value[index].label
    zip.file(`spectrogram-${index}-${label}.png`, base64, { base64: true })
  }
  const blob = await zip.generateAsync({ type: 'blob' })
  saveAs(blob, 'spectrograms.zip')
}

// === 放大预览 ===
function openPreview(index) {
  previewIndex.value = index
}
function closePreview() {
  previewIndex.value = null
}
</script>


<style scoped>
.spectrogram-wrapper {
  max-width: 1300px;
  margin: 0 auto;
  padding: 24px;
  text-align: center;
}
.main-title {
  font-size: 24px;
  font-weight: bold;
  margin-bottom: 24px;
}
.upload-card {
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 12px;
  padding: 24px 32px;
  margin: 0 auto 24px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  max-width: 900px;
}

.filename {
  font-size: 13px;
  color: #444;
  margin-top: 8px;
  display: block;
  text-align: center;
}


.block-btn {
  display: block;
  width: 100%;
  font-size: 16px;
  padding: 12px 20px;
  border-radius: 8px;
  font-weight: 600;
  margin-bottom: 12px;
  border: none;
}

.btn-column {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  justify-content: center;
  align-items: center;
}

.long-btn {
  font-size: 16px;
  padding: 10px 0;
  border-radius: 8px;
  font-weight: 600;
  width: 100%;
  border: none;
}

.btn-row {
  display: flex;
  gap: 16px;
  justify-content: center;
  align-items: center;
  flex-wrap: wrap;
}
.btn-wide {
  flex: 1 1 300px;
  font-size: 16px;
  padding: 12px 20px;
  border-radius: 8px;
  font-weight: 600;
  text-align: center;
  transition: 0.2s;
  border: none;
}


.action-btn {
  width: 80%;
  font-size: 15px;
  padding: 10px 0;
  margin: 0 auto 12px;
  display: block;
  border-radius: 8px;
  font-weight: 600;
  border: none;
  transition: 0.2s;
}
.select-btn {
  background-color: #f0f0f0;
  color: #333;
  border: 1px solid #ccc;
}
.analyze-btn {
  background-color: #007bff;
  color: #fff;
}
.analyze-btn:disabled {
  background-color: #a0c0e0;
  cursor: not-allowed;
}
.export-btn {
  font-size: 15px;
  padding: 8px 32px;
  border-radius: 6px;
}

.grid-gallery {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 20px;
}
.chart-box {
  background: #fafafa;
  border: 1px solid #ddd;
  border-radius: 10px;
  padding: 14px;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
}
canvas.previewable {
  cursor: zoom-in;
  transition: transform 0.2s;
}
canvas.previewable:hover {
  transform: scale(1.02);
}
.label {
  margin-top: 6px;
  font-weight: 600;
}
.summary {
  font-size: 13px;
  color: #555;
  margin: 4px 0;
}
.overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  z-index: 999;
  display: flex;
  align-items: center;
  justify-content: center;
}
.preview-dialog {
  background: white;
  padding: 24px;
  border-radius: 12px;
  box-shadow: 0 0 30px rgba(0, 0, 0, 0.2);
  text-align: center;
}
</style>
