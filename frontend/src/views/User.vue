<template>
  <div class="user-wrapper">
    <h2 class="main-title">用户管理</h2>
    <p class="desc">管理平台用户账号与权限</p>

    <div class="card-box">
      <div class="d-flex justify-content-between align-items-center mb-3">
        <button class="btn btn-primary btn-sm">添加用户</button>
        <button class="btn btn-outline-secondary btn-sm">角色权限设置</button>
      </div>

      <div class="table-responsive">
        <table class="table table-hover table-bordered table-sm">
          <thead class="table-light">
            <tr>
              <th>用户 ID</th>
              <th>用户名</th>
              <th>角色</th>
              <th>邮箱</th>
              <th>状态</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="user in mockUsers" :key="user.id">
              <td>{{ user.id }}</td>
              <td>{{ user.username }}</td>
              <td>{{ user.role }}</td>
              <td>{{ user.email }}</td>
              <td>
                <span :class="['badge', user.active ? 'bg-success' : 'bg-secondary']">
                  {{ user.active ? '启用中' : '已停用' }}
                </span>
              </td>
              <td>
                <button class="btn btn-sm btn-outline-primary me-2">编辑</button>
                <button class="btn btn-sm btn-outline-danger">删除</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <p class="note mt-3">提示：以上为模拟用户数据，功能接口尚未接入。</p>
    </div>
  </div>
</template>

<script setup>
const mockUsers = [
  { id: 1, username: 'admin', role: '超级管理员', email: 'admin@example.com', active: true },
  { id: 2, username: 'user001', role: '普通用户', email: 'user001@example.com', active: true },
  { id: 3, username: 'guest', role: '访客', email: 'guest@example.com', active: false }
]
</script>

<style scoped>
.user-wrapper {
  max-width: 960px;
  margin: 0 auto;
  text-align: center;
}
.main-title {
  font-size: 22px;
  font-weight: 600;
  color: #222;
  margin-bottom: 6px;
}
.desc {
  font-size: 14px;
  color: #888;
  margin-bottom: 24px;
}
.card-box {
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.04);
  padding: 30px;
  text-align: left;
}
.note {
  font-size: 13px;
  color: #999;
}
</style>
