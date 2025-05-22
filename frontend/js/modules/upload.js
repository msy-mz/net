// Filename: upload.js
// Path: frontend/js/modules/upload.js
// Description: 绑定上传表单事件，调用后端推理接口，反馈推理状态
// Author: msy
// Date: 2025

import { showModule } from '../main.js';
import { initFigures } from './visual.js';

export function initUpload() {
  const form = document.getElementById('upload-form');
  const resultCard = document.getElementById('upload-result');
  const statusBox = document.getElementById('upload-status');
  const detailBox = document.getElementById('upload-detail');

  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById('pcap-file');
    const file = fileInput.files[0];
    if (!file) {
      alert('请选择一个 .pcap 文件');
      return;
    }

    // UI 初始化
    resultCard.classList.remove('d-none');
    statusBox.textContent = '上传中...';
    detailBox.innerHTML = '';

    const formData = new FormData();
    formData.append('file', file);

    try {
      const uploadResp = await fetch('/upload_pcap', {
        method: 'POST',
        body: formData
      });
      const uploadData = await uploadResp.json();

      if (uploadData.status !== 'success') {
        statusBox.textContent = '上传失败';
        detailBox.textContent = uploadData.message || '未知错误';
        return;
      }

      statusBox.textContent = '上传成功，正在推理...';
      const inferResp = await fetch('/run_infer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pcap_path: uploadData.pcap_path })
      });
      const inferData = await inferResp.json();

      statusBox.textContent = inferData.message;
      detailBox.textContent = '';

      // 跳转到可视化模块
      showModule("visual");
      setTimeout(initFigures, 500);

    } catch (err) {
      console.error('上传或推理失败', err);
      statusBox.textContent = '❌ 上传或分析失败';
      detailBox.innerHTML = `<pre>${err}</pre>`;
    }
  });
}
