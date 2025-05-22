// Filename: monitor.js
// Path: frontend/js/modules/monitor.js

import Chart from 'https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.esm.js';

let chart = null;
let dataBuffer = [];
let isRunning = true;

function randomInt(min, max) {
  return Math.floor(Math.random() * (max - min) + min);
}

function updateStats() {
  const conn = randomInt(50, 100);
  const abn = randomInt(5, 30);
  const total = randomInt(1000, 5000);

  document.getElementById('conn-count').textContent = conn;
  document.getElementById('abn-ratio').textContent = `${Math.round((abn / conn) * 100)}%`;
  document.getElementById('packet-total').textContent = total;
}

function updateChart() {
  const now = new Date().toLocaleTimeString().slice(3, 8);
  const value = randomInt(20, 120);
  dataBuffer.push({ t: now, v: value });

  if (dataBuffer.length > 20) dataBuffer.shift();

  chart.data.labels = dataBuffer.map(d => d.t);
  chart.data.datasets[0].data = dataBuffer.map(d => d.v);
  chart.update();
}

function bindToggle() {
  const btn = document.getElementById('toggle-btn');
  const statusText = document.getElementById('status-text');

  btn.addEventListener('click', () => {
    isRunning = !isRunning;
    statusText.textContent = isRunning ? '监听中' : '已暂停';
    statusText.classList.toggle('text-danger', !isRunning);
    btn.textContent = isRunning ? '暂停监听' : '恢复监听';
  });
}

export function initMonitor() {
  const ctx = document.getElementById('realtime-chart')?.getContext('2d');
  if (!ctx) return;

  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: '异常流量数',
        data: [],
        borderColor: '#f00',
        backgroundColor: 'rgba(255,0,0,0.2)',
        tension: 0.4,
        fill: true
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: {
          beginAtZero: true,
          suggestedMax: 150
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });

  bindToggle();

  // 定时更新数据（模拟监听）
  setInterval(() => {
    if (isRunning) {
      updateChart();
      updateStats();
    }
  }, 1000);
}
