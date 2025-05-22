// Filename: upload.js
// Path: frontend/js/modules/upload.js

export function initUpload() {
  const form = document.getElementById('upload-form');
  const resultCard = document.getElementById('upload-result');
  const statusBox = document.getElementById('upload-status');
  const detailBox = document.getElementById('upload-detail');

  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById('pcap-file');
    const modelSelect = document.getElementById('model-select');
    const file = fileInput.files[0];
    const model = modelSelect.value;

    if (!file) {
      alert('请选择一个 .pcap 文件');
      return;
    }

    // UI 初始化
    resultCard.classList.remove('d-none');
    statusBox.textContent = '上传中...';
    detailBox.innerHTML = '';

    const formData = new FormData();
    formData.append('pcap', file);
    formData.append('model', model);

    try {
      // ⚠️ 替换成你的真实 Flask 接口
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      const data = await res.json();

      statusBox.textContent = '分析完成';
      detailBox.innerHTML = `
        <p><strong>预测标签：</strong> ${data.label}</p>
        <p><strong>置信度：</strong> ${data.confidence}</p>
      `;
    } catch (err) {
      console.error('上传失败', err);
      statusBox.textContent = '❌ 上传或分析失败';
      detailBox.innerHTML = `<pre>${err}</pre>`;
    }
  });
}
