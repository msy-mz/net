// Filename: src/stores/userStore.js
// Author: msy

import { defineStore } from 'pinia'

export const useUserStore = defineStore('user', {
  state: () => ({
    users: [
      { id: 1, username: 'admin', role: '超级管理员', email: 'admin@example.com', active: true },
      { id: 2, username: 'user001', role: '普通用户', email: 'user001@example.com', active: true },
      { id: 3, username: 'guest', role: '访客', email: 'guest@example.com', active: false }
    ]
  }),
  actions: {
    addUser(user) {
      this.users.push(user)
    },
    updateUser(user) {
      const i = this.users.findIndex(u => u.id === user.id)
      if (i !== -1) this.users[i] = { ...user }
    },
    deleteUser(id) {
      this.users = this.users.filter(u => u.id !== id)
    }
  }
})
