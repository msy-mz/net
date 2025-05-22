// Filename: visual.js
// Path: frontend/js/modules/visual.js
// Description: 图像展示模块（专业排版 + 统计摘要 + 中文标识）
// Author: msy
// Date: 2025

export async function initFigures() {
  const vis = document.getElementById("vis-area");
  if (!vis) return;

  // 从 summary.json 获取统计数据
  const summaryResp = await fetch('/result/inf/infer/summary.json');
  const summary = await summaryResp.json();
  const total = summary.total_flows;
  const abnormal = summary.abnormal_flows;

  vis.innerHTML = `
    <h4 class="mb-3">📊 推理分析报告概览</h4>
    <div class="alert alert-info">
      <p class="mb-1">总流量数：<strong>${total}</strong></p>
      <p class="mb-1">异常流量数：<strong>${abnormal}</strong></p>
      <p class="mb-0">分类分布与频谱特征如下：</p>
    </div>

    <div class="row">
      <div class="col-md-6 mb-4">
        <div class="card p-3 shadow-sm">
          <h6 class="mb-2">分类分布图</h6>
          <img src="/result/inf/infer/label_distribution.png" class="img-fluid rounded" />
          <p class="text-muted mt-2 small">展示各类流量在推理结果中的数量分布情况。</p>
        </div>
      </div>
      <div class="col-md-6 mb-4">
        <div class="card p-3 shadow-sm">
          <h6 class="mb-2">PCA 聚类图</h6>
          <img src="/result/inf/infer/feature_vis/global_spectrum/pca_spectrum_clusters.png" class="img-fluid rounded" />
          <p class="text-muted mt-2 small">将高维频谱特征降维至二维空间以展示聚类效果。</p>
        </div>
      </div>
    </div>

    <div class="row">
      <div class="col-md-6 mb-4">
        <div class="card p-3 shadow-sm">
          <h6 class="mb-2">正常 vs 异常频谱图</h6>
          <img src="/result/inf/infer/feature_vis/global_spectrum/normal_vs_abnormal_spectrum.png" class="img-fluid rounded" />
          <p class="text-muted mt-2 small">分别绘制正常与异常流量的平均频谱曲线，用于对比差异。</p>
        </div>
      </div>
      <div class="col-md-6 mb-4">
        <div class="card p-3 shadow-sm">
          <h6 class="mb-2">谱熵分布图</h6>
          <img src="/result/inf/infer/feature_vis/global_spectrum/spectral_entropy_hist.png" class="img-fluid rounded" />
          <p class="text-muted mt-2 small">谱熵衡量频谱平坦程度，可用于辅助判断异常流。</p>
        </div>
      </div>
    </div>
  `;
}
