// Filename: main.js
// Path: frontend/js/main.js
// Description: 模块导航切换控制 + 顶部/侧边栏加载 + 动态加载模块 HTML 与 JS
// Author: msy
// Date: 2025

const DEFAULT_MODULE_ID = 'dashboard';

const moduleMap = {
  dashboard: '系统总览',
  monitor: '实时监听分析',
  upload: '上传文件分析',
  visual: '可视化分析',
  log: '监听日志报告'
};

// ✅ 用相对路径写法，确保兼容打包与部署
const moduleInit = {
  dashboard: async () => {
    try {
      const m = await import('./modules/dashboard.js');
      m.initDashboard && m.initDashboard();
    } catch (e) {
      console.warn('[dashboard] 未提供 JS 模块', e);
    }
  },
  monitor: async () => {
    try {
      const m = await import('./modules/monitor.js');
      m.initMonitor && m.initMonitor();
    } catch (e) {
      console.warn('[monitor] 未提供 JS 模块', e);
    }
  },
  upload: async () => {
    try {
      const m = await import('./modules/upload.js');
      m.initUpload && m.initUpload();
    } catch (e) {
      console.warn('[upload] 未提供 JS 模块', e);
    }
  },
  visual: async () => {
    try {
      const m = await import('./modules/visual.js');
      m.initFigures && m.initFigures();
    } catch (e) {
      console.warn('[visual] 未提供 JS 模块', e);
    }
  },
  log: async () => {
    try {
      const m = await import('./modules/log.js');
      m.initLog && m.initLog();
    } catch (e) {
      console.warn('[log] 未提供 JS 模块', e);
    }
  }
};

const loaded = new Set();

// 加载模块 HTML 并初始化 JS
export async function loadModule(id) {
  const section = document.getElementById(id);
  if (!section) return;

  if (!loaded.has(id)) {
    try {
      const res = await fetch(`/components/${id}.html`);
      const html = await res.text();
      section.innerHTML = html;
      loaded.add(id);
      if (moduleInit[id]) await moduleInit[id]();
    } catch (err) {
      section.innerHTML = `<p class="text-danger">模块加载失败：${id}</p>`;
      console.error(err);
    }
  }

  // 激活模块显示
  document.querySelectorAll('section.module').forEach(s => s.classList.remove('active'));
  section.classList.add('active');

  // 同步导航高亮
  document.querySelectorAll('.sidebar a').forEach(link => {
    link.classList.toggle('active', link.getAttribute('data-module') === id);
  });

  // 更新面包屑
  const title = moduleMap[id] || id;
  document.getElementById('breadcrumb-box').innerHTML = `<span class="breadcrumb-item active">${title}</span>`;
}

// 加载 sidebar.html 和 topbar.html
async function loadFrameComponent(targetId, filePath) {
  const target = document.getElementById(targetId);
  if (!target) return;
  try {
    const res = await fetch(`/${filePath}`);
    const html = await res.text();
    target.innerHTML = html;
  } catch (err) {
    target.innerHTML = `<p class="text-danger">加载失败：${filePath}</p>`;
    console.error(`[错误] 组件加载失败: ${filePath}`, err);
  }
}

// 绑定导航点击事件
function initSidebarNavigation() {
  document.querySelectorAll('.sidebar a').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const id = link.getAttribute('data-module');
      if (id) loadModule(id);
    });
  });
}

// 初始化页面框架与默认模块
document.addEventListener('DOMContentLoaded', async () => {
  await loadFrameComponent('sidebar-box', 'components/sidebar.html');
  await loadFrameComponent('topbar-box', 'components/topbar.html');

  setInterval(() => {
    const now = new Date();
    const timeBox = document.getElementById("time-box");
    if (timeBox) timeBox.textContent = now.toLocaleString();
  }, 1000);

  initSidebarNavigation();
  loadModule(DEFAULT_MODULE_ID);
});

// main.js 底部添加
export { loadModule as showModule };
