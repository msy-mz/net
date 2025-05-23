// Filename: upload.js
// Path: frontend/js/modules/upload.js
// Description: 多模型支持版本（附带模型名传参）
// Author: msy
// Date: 2025

import { showModule } from '../main.js';
import { initFigures } from './visual.js';

export function initUpload() {
  const form = document.getElementById('upload-form');
  const resultCard = document.getElementById('upload-result');
  const statusBox = document.getElementById('upload-status');
  const detailBox = document.getElementById('upload-detail');
  const submitBtn = form.querySelector('button[type="submit"]');
  const modelSelect = document.getElementById('model-select');

  if (!form || !modelSelect) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById('pcap-file');
    const files = fileInput.files;
    const model = modelSelect.value;

    if (!files || files.length === 0) {
      alert('请选择至少一个 .pcap 文件');
      return;
    }

    resultCard.classList.remove('d-none');
    statusBox.textContent = '上传中...';
    detailBox.innerHTML = '';
    submitBtn.disabled = true;

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('file', files[i]);
    }

    try {
      const uploadResp = await fetch('/upload_pcap', {
        method: 'POST',
        body: formData
      });

      const uploadData = await uploadResp.json();
      console.log('[upload] 上传响应:', uploadData);

      if (uploadData.status !== 'success') {
        statusBox.textContent = '上传失败';
        detailBox.textContent = uploadData.message || '未知错误';
        submitBtn.disabled = false;
        return;
      }

      statusBox.textContent = '上传成功，正在推理...';

      const inferResp = await fetch('/run_infer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pcap_paths: uploadData.pcap_paths,
          model: model
        })
      });

      const inferData = await inferResp.json();
      statusBox.textContent = inferData.message;
      detailBox.textContent = '';
      submitBtn.disabled = false;

      setTimeout(() => {
        showModule("visual");
        initFigures();
      }, 1000);

    } catch (err) {
      console.error('[upload] 异常:', err);
      statusBox.textContent = '上传或分析失败';
      detailBox.innerHTML = `<pre>${err}</pre>`;
      submitBtn.disabled = false;
    }
  });
}
