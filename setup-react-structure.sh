#!/bin/bash

# Filename: setup-react-structure.sh
# Path: ./setup-react-structure.sh
# Description: 创建 React 前端目录结构，生成页面组件与路由模块
# Author: msy
# Date: 2025

set -e

echo "✅ 创建 React 项目结构..."

# 进入 src 目录
cd src

# 创建目录结构
mkdir -p pages components assets router

# 创建页面组件
pages=(Dashboard Monitor Upload Visual Log NistTest)
for page in "${pages[@]}"; do
  cat > "pages/${page}.jsx" <<EOF
// Filename: ${page}.jsx
// Path: src/pages/${page}.jsx
// Description: ${page} 页面组件
// Author: msy
// Date: 2025

import React from 'react'

function ${page}() {
  return (
    <div>
      <h1>${page} 页面</h1>
      <p>这是 ${page} 模块。</p>
    </div>
  )
}

export default ${page}
EOF
done

# 创建 App.jsx
cat > App.jsx <<EOF
// Filename: App.jsx
// Path: src/App.jsx
// Description: 应用主组件，配置路由和导航
// Author: msy
// Date: 2025

import React from 'react'
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Monitor from './pages/Monitor'
import Upload from './pages/Upload'
import Visual from './pages/Visual'
import Log from './pages/Log'
import NistTest from './pages/NistTest'

function App() {
  return (
    <Router>
      <nav style={{ margin: '20px', display: 'flex', gap: '20px' }}>
        <Link to="/">Dashboard</Link>
        <Link to="/monitor">Monitor</Link>
        <Link to="/upload">Upload</Link>
        <Link to="/visual">Visual</Link>
        <Link to="/log">Log</Link>
        <Link to="/nist-test">NistTest</Link>
      </nav>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/monitor" element={<Monitor />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/visual" element={<Visual />} />
        <Route path="/log" element={<Log />} />
        <Route path="/nist-test" element={<NistTest />} />
      </Routes>
    </Router>
  )
}

export default App
EOF

echo "✅ 结构生成完成，可使用 npm run dev 启动开发服务器"
