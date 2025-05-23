// Filename: monitor.js
// Path: frontend/js/modules/monitor.js
// Description: 实时监听监控模块（控制 + 状态 +图表）
// Author: msy
// Date: 2025

const Chart = window.Chart;

let chart = null;
let dataBuffer = [];
let isRunning = false;  // 由状态接口决定

async function fetchRealtimeStatus() {
  try {
    const res = await fetch('/realtime/status');

    // 确保 HTTP 状态码为 200，并且响应类型是 JSON
    const contentType = res.headers.get('content-type') || '';
    if (!res.ok || !contentType.includes('application/json')) {
      throw new Error(`状态请求失败，HTTP ${res.status}`);
    }

    const data = await res.json();
    return data.running === true;
  } catch (err) {
    console.warn('[状态检查失败]', err);
    return false;
  }
}


async function fetchRealtimeStatus() {
  try {
    const res = await fetch('/realtime/status');
    const data = await res.json();
    return data.running === true;
  } catch (err) {
    console.warn('[状态检查失败]', err);
    return false;
  }
}

async function startRealtimeBackend() {
  const res = await fetch('/realtime/start', { method: 'POST' });
  const data = await res.json();
  console.log('[启动结果]', data);
  return data.status === 'started' || data.status === 'running';
}

async function stopRealtimeBackend() {
  const res = await fetch('/realtime/stop', { method: 'POST' });
  const data = await res.json();
  console.log('[停止结果]', data);
  return data.status === 'stopped';
}

async function updateStatsAndChart() {
  const data = await fetchRealtimeData();
  if (!data || !Array.isArray(data)) return;

  const connSet = new Set();
  let abnCount = 0;
  let totalLength = 0;

  data.forEach(item => {
    connSet.add(`${item.src_ip}-${item.dst_ip}-${item.dst_port}`);
    if (item.is_abnormal) abnCount++;
    totalLength += item.length || 0;
  });

  const conn = connSet.size;
  const abn = abnCount;
  const total = totalLength;

  document.getElementById('conn-count').textContent = conn;
  document.getElementById('abn-ratio').textContent = conn ? `${Math.round((abn / conn) * 100)}%` : '0%';
  document.getElementById('packet-total').textContent = total;

  const now = new Date().toLocaleTimeString().slice(3, 8);
  dataBuffer.push({ t: now, v: abn });
  if (dataBuffer.length > 20) dataBuffer.shift();

  chart.data.labels = dataBuffer.map(d => d.t);
  chart.data.datasets[0].data = dataBuffer.map(d => d.v);
  chart.update();

  updateLogTable(data.slice(-5).reverse());
}

function updateLogTable(recentLogs) {
  const tableBody = document.getElementById('log-table-body');
  if (!tableBody) return;

  tableBody.innerHTML = '';
  for (const item of recentLogs) {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${item.timestamp || '--'}</td>
      <td>${item.src_ip}:${item.src_port}</td>
      <td>${item.dst_ip}:${item.dst_port}</td>
      <td>${item.label}</td>
      <td>${item.is_abnormal ? '<span class="text-danger">是</span>' : '否'}</td>
    `;
    tableBody.appendChild(row);
  }
}

function bindToggle() {
  const btn = document.getElementById('toggle-btn');
  const statusText = document.getElementById('status-text');

  btn.addEventListener('click', async () => {
    if (isRunning) {
      const stopped = await stopRealtimeBackend();
      if (stopped) {
        isRunning = false;
        updateStatusUI();
      }
    } else {
      const started = await startRealtimeBackend();
      if (started) {
        isRunning = true;
        updateStatusUI();
      }
    }
  });
}

function updateStatusUI() {
  const statusText = document.getElementById('status-text');
  const btn = document.getElementById('toggle-btn');
  statusText.textContent = isRunning ? '监听中' : '已暂停';
  statusText.classList.toggle('text-danger', !isRunning);
  btn.textContent = isRunning ? '暂停监听' : '启动监听';
}

export async function initMonitor() {
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
          suggestedMax: 100
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });

  // 检查初始状态
  isRunning = await fetchRealtimeStatus();
  updateStatusUI();

  bindToggle();

  setInterval(async () => {
    const running = await fetchRealtimeStatus();
    isRunning = running;
    updateStatusUI();

    if (isRunning) {
      updateStatsAndChart();
    }
  }, 2000);
}
