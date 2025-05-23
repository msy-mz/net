<template>
  <div class="user-wrapper">
    <h2 class="main-title">用户管理</h2>
    <p class="desc">管理平台用户账号与权限</p>

    <div class="card-box">
      <div class="d-flex justify-content-between align-items-center mb-3">
        <button class="btn btn-primary btn-sm" @click="openAddDialog">添加用户</button>
        <button class="btn btn-outline-secondary btn-sm" @click="showRoleSetting">角色权限设置</button>
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
            <tr v-for="user in store.users" :key="user.id">
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
                <button class="btn btn-sm btn-outline-primary me-2" @click="edit(user)">编辑</button>
                <button class="btn btn-sm btn-outline-danger me-2" @click="remove(user.id)">删除</button>
                <button class="btn btn-sm btn-outline-warning" @click="resetPassword(user)">重置密码</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- 用户编辑弹窗 -->
    <el-dialog v-model="showEditDialog" :title="isNew ? '添加用户' : '编辑用户'" width="400px">
      <el-form :model="editUser" label-width="80px">
        <el-form-item label="用户名">
          <el-input v-model="editUser.username" />
        </el-form-item>
        <el-form-item label="邮箱">
          <el-input v-model="editUser.email" />
        </el-form-item>
        <el-form-item label="角色">
          <el-select v-model="editUser.role" placeholder="选择角色">
            <el-option label="超级管理员" value="超级管理员" />
            <el-option label="普通用户" value="普通用户" />
            <el-option label="访客" value="访客" />
          </el-select>
        </el-form-item>
        <el-form-item label="状态">
          <el-switch v-model="editUser.active" active-text="启用" inactive-text="停用" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showEditDialog = false">取消</el-button>
        <el-button type="primary" @click="saveUser">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useUserStore } from '@/stores/userStore'

const store = useUserStore()
const showEditDialog = ref(false)
const editUser = ref({})
const isNew = ref(false)

function openAddDialog() {
  isNew.value = true
  editUser.value = {
    id: Date.now(),
    username: '',
    email: '',
    role: '普通用户',
    active: true
  }
  showEditDialog.value = true
}

function edit(user) {
  isNew.value = false
  editUser.value = { ...user }
  showEditDialog.value = true
}

function saveUser() {
  if (isNew.value) {
    store.addUser({ ...editUser.value })
    ElMessage.success('用户已添加')
  } else {
    store.updateUser({ ...editUser.value })
    ElMessage.success('用户信息已更新')
  }
  showEditDialog.value = false
}

function remove(id) {
  ElMessageBox.confirm('确认删除该用户吗？', '提示', {
    confirmButtonText: '确认',
    cancelButtonText: '取消',
    type: 'warning'
  }).then(() => {
    store.deleteUser(id)
    ElMessage.success('删除成功')
  }).catch(() => {})
}

function resetPassword(user) {
  ElMessageBox.confirm(`确认重置 ${user.username} 的密码？`, '提示', {
    confirmButtonText: '确认',
    cancelButtonText: '取消',
    type: 'info'
  }).then(() => {
    ElMessage.success(`已重置 ${user.username} 的密码（模拟）`)
  }).catch(() => {})
}

function showRoleSetting() {
  ElMessage.info('角色权限设置功能尚未实现（示例占位）')
}
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
</style>
