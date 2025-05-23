<!-- Filename: Dashboard.vue -->
<!-- Path: frontend/views/Dashboard.vue -->
<!-- Description: 系统总览页面 -->
<!-- Author: msy -->
<!-- Date: 2025 -->

<template>
  <div class="container mt-4 text-center">
    <h2 class="fw-bold mb-2">FT-IDNet++ 网络流量监测平台</h2>
    <p class="text-muted mb-4 fs-6">基于深度学习的网络流量智能分析系统</p>

    <div class="text-start">
      <QuickEntry />
      <div class="row g-4 mb-4">
        <StatCard />
      </div>
      <TrendChart />
    </div>
  </div>
</template>



<script setup>
// 导入子组件
import QuickEntry from '../components/QuickEntry.vue'
import StatCard from '../components/StatCard.vue'
import TrendChart from '../components/TrendChart.vue'
</script>
