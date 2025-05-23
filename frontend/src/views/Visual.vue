<template>
  <div class="visual-wrapper">
    <h2 class="main-title">可视化分析</h2>
    <!-- <p class="desc">展示模型输出的特征降维与聚类图像</p> -->

    <div class="grid">
      <div class="card-box" v-for="(url, index) in imageUrls" :key="index">
        <div class="image-wrap">
          <img :src="url" class="img-fluid" alt="聚类图像" />
        </div>
        <div class="caption">{{ getFilename(url) }}</div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const imageUrls = ref([
  'http://localhost:8888/result/file/inf/infer/feature_vis/tsne.png',
  'http://localhost:8888/result/file/inf/infer/feature_vis/cluster.png'
])

const getFilename = (url) => url.split('/').pop()
</script>

<style scoped>
.visual-wrapper {
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
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 24px;
}
.card-box {
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
  padding: 16px;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.image-wrap {
  width: 100%;
  border-radius: 8px;
  overflow: hidden;
}
.img-fluid {
  width: 100%;
  height: auto;
  display: block;
}
.caption {
  margin-top: 12px;
  font-size: 13px;
  color: #666;
}
</style>
