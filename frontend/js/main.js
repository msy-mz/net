// Filename: main.js
// Path: frontend/js/main.js
// Description: 模块导航逻辑控制 + 动态 HTML 懒加载 + JS 初始化绑定（支持 upload/visual 模块）
// Author: msy
// Date: 2025

// 默认激活模块 ID
const DEFAULT_MODULE_ID = 'dashboard';

// 模块 ID 到名称的映射，用于面包屑与标题
const moduleMap = {
  dashboard: '系统总览',
  monitor: '实时监听分析',
  upload: '上传文件分析',
  visual: '可视化分析',
  log: '监听日志报告'
};

// 各模块 JS 初始化函数（使用 import 动态加载）
const moduleInit = {
  dashboard: async () => {
    try {
      const m = await import('./modules/dashboard.js');
      m.initDashboard && m.initDashboard();
    } catch (e) {
      console.warn('[dashboard] 未提供 JS 模块，已跳过初始化');
    }
  },
  monitor: async () => {
    try {
      const m = await import('./modules/monitor.js');
      m.initMonitor && m.initMonitor();
    } catch (e) {
      console.warn('[monitor] 未提供 JS 模块，已跳过初始化');
    }
  },
  upload: async () => {
    try {
      const m = await import('./modules/upload.js');
      m.initUpload && m.initUpload();
    } catch (e) {
      console.warn('[upload] 未提供 JS 模块，已跳过初始化');
    }
  },
  visual: async () => {
    try {
      const m = await import('./modules/visual.js');
      m.initFigures && m.initFigures();
    } catch (e) {
      console.warn('[visual] 未提供 JS 模块，已跳过初始化');
    }
  },
  log: async () => {
    try {
      const m = await import('./modules/log.js');
      m.initLog && m.initLog();
    } catch (e) {
      console.warn('[log] 未提供 JS 模块，已跳过初始化');
    }
  }
};

// 已加载模块缓存集合，避免重复加载
const loaded = new Set();

/**
 * 加载模块 HTML 内容并激活模块
 * @param {string} id - 模块 ID
 */
export async function loadModule(id) {
  const section = document.getElementById(id);
  if (!section) return;

  // 首次加载 HTML 结构
  if (!loaded.has(id)) {
    try {
      const res = await fetch(`/components/${id}.html`);
      const html = await res.text();
      section.innerHTML = html;
      loaded.add(id);

      // 模块 JS 初始化调用（如果存在）
      const initFunc = moduleInit[id];
      if (initFunc) await initFunc();
    } catch (err) {
      section.innerHTML = `<p class="text-danger">模块加载失败：${id}</p>`;
      console.error(`[加载失败] /components/${id}.html`, err);
    }
  }

  // 激活当前模块，隐藏其他模块
  document.querySelectorAll('section.module').forEach(s => s.classList.remove('active'));
  section.classList.add('active');

  // 更新面包屑导航
  const title = moduleMap[id] || id;
  document.getElementById('breadcrumb-box').innerHTML = `<span class="breadcrumb-item active">${title}</span>`;
}

/**
 * 初始化左侧导航栏点击事件
 */
function initSidebarNavigation() {
  const links = document.querySelectorAll('.sidebar a');
  links.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const id = link.getAttribute('data-module');
      if (!id) return;

      // 设置当前高亮
      links.forEach(l => l.classList.remove('active'));
      link.classList.add('active');

      // 加载对应模块
      loadModule(id);
    });
  });
}

// 页面加载完成后执行默认模块激活
document.addEventListener('DOMContentLoaded', () => {
  initSidebarNavigation();
  loadModule(DEFAULT_MODULE_ID);
});
