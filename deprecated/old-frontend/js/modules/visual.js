// Filename: visual.js
// Path: frontend/js/modules/visual.js
// Description: 图像展示模块（美观卡片式 + 统计摘要 + 自动滚动）
// Author: msy
// Date: 2025

export async function initFigures() {
  const vis = document.getElementById("figures-box");
  if (!vis) return;

  try {
    const summaryResp = await fetch('/result/inf/infer/summary.json');
    const summary = await summaryResp.json();
    const total = summary.total_flows;
    const abnormal = summary.abnormal_flows;

    vis.innerHTML = `
      <div class="card p-4 shadow-sm">
        <h4 class="mb-3">推理分析报告概览</h4>
        <div class="alert alert-info">
          <p class="mb-1">总流量数：<strong>${total}</strong></p>
          <p class="mb-1">异常流量数：<strong>${abnormal}</strong></p>
          <p class="mb-0">以下为详细图像化分析结果：</p>
        </div>

        <div class="row">
          <div class="col-md-6 mb-4">
            <div class="card p-3 shadow-sm h-100">
              <h6>分类分布图</h6>
              <img src="/result/inf/infer/label_distribution.png" class="img-fluid rounded" />
              <p class="text-muted small mt-2">展示各类网络流在推理输出中的数量分布。</p>
            </div>
          </div>
          <div class="col-md-6 mb-4">
            <div class="card p-3 shadow-sm h-100">
              <h6>PCA 聚类图</h6>
              <img src="/result/inf/infer/feature_vis/global_spectrum/pca_spectrum_clusters.png" class="img-fluid rounded" />
              <p class="text-muted small mt-2">展示所有频谱特征经降维后的聚类可视化。</p>
            </div>
          </div>
        </div>

        <div class="row">
          <div class="col-md-6 mb-4">
            <div class="card p-3 shadow-sm h-100">
              <h6>正常 vs 异常频谱</h6>
              <img src="/result/inf/infer/feature_vis/global_spectrum/normal_vs_abnormal_spectrum.png" class="img-fluid rounded" />
              <p class="text-muted small mt-2">展示正常流与异常流在频谱维度的平均差异。</p>
            </div>
          </div>
          <div class="col-md-6 mb-4">
            <div class="card p-3 shadow-sm h-100">
              <h6>谱熵分布直方图</h6>
              <img src="/result/inf/infer/feature_vis/global_spectrum/spectral_entropy_hist.png" class="img-fluid rounded" />
              <p class="text-muted small mt-2">谱熵衡量频谱信息的平坦性，辅助识别流类型复杂度。</p>
            </div>
          </div>
        </div>
      </div>
    `;

    // 自动滚动至图像区
    vis.scrollIntoView({ behavior: 'smooth' });

  } catch (err) {
    console.error("可视化模块加载失败", err);
    vis.innerHTML = `
      <div class="card p-4 shadow-sm">
        <div class="text-danger">❌ 加载可视化数据失败，请确认是否已有推理输出结果。</div>
      </div>
    `;
  }
}
