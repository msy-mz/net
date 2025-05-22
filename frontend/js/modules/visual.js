// Filename: visual.js
// Path: frontend/js/modules/visual.js

import Chart from 'https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.esm.js';

let chart = null;

function randomClusterPoints(num, label, color) {
  return Array.from({ length: num }, () => ({
    x: Math.random() * 10 + (label === 'malicious' ? 0 : 10),
    y: Math.random() * 10,
    label,
    backgroundColor: color
  }));
}

function drawClusterChart(model, source) {
  const normal = randomClusterPoints(40, 'normal', 'rgba(54,162,235,0.6)');
  const malicious = randomClusterPoints(30, 'malicious', 'rgba(255,99,132,0.7)');
  const data = [...normal, ...malicious];

  const ctx = document.getElementById('cluster-chart')?.getContext('2d');
  if (!ctx) return;

  if (chart) {
    chart.destroy();
  }

  chart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: '正常流量',
          data: data.filter(p => p.label === 'normal'),
          backgroundColor: 'rgba(54,162,235,0.6)'
        },
        {
          label: '恶意流量',
          data: data.filter(p => p.label === 'malicious'),
          backgroundColor: 'rgba(255,99,132,0.7)'
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'top' },
        title: {
          display: true,
          text: `模型：${model.toUpperCase()}｜来源：${source === 'upload' ? '上传' : '监听'}`
        }
      },
      scales: {
        x: { beginAtZero: true },
        y: { beginAtZero: true }
      }
    }
  });
}

export function initFigures() {
  const modelSelect = document.getElementById('global-model-select');
  const sourceSelect = document.getElementById('source-select');

  function render() {
    const model = modelSelect.value;
    const source = sourceSelect.value;
    drawClusterChart(model, source);
  }

  modelSelect.addEventListener('change', render);
  sourceSelect.addEventListener('change', render);

  render();  // 初次渲染
}
