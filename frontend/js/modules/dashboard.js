// Filename: dashboard.js
// Path: frontend/js/modules/dashboard.js
// Description: 
// Author: msy
// Date: 2025


const Chart = window.Chart;


export function initDashboard() {
  // 模拟异常趋势图
  const ctx = document.getElementById('trend-chart')?.getContext('2d');
  if (ctx) {
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: Array.from({ length: 10 }, (_, i) => `${8 + i}:00`),
        datasets: [{
          label: '异常请求数',
          data: Array.from({ length: 10 }, () => Math.floor(Math.random() * 50 + 10)),
          backgroundColor: 'rgba(255, 99, 132, 0.5)'
        }]
      },
      options: {
        plugins: {
          legend: { display: false }
        },
        scales: {
          y: { beginAtZero: true }
        }
      }
    });
  }

  // 快速入口卡片点击跳转
  document.querySelectorAll('.quick-entry').forEach(card => {
    card.addEventListener('click', () => {
      const id = card.dataset.module;
      const link = document.querySelector(`.sidebar a[data-module="${id}"]`);
      if (link) link.click();  // 触发导航切换
    });
  });
}
