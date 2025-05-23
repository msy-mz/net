import { defineStore } from 'pinia'

export const useListenStore = defineStore('listen', {
  state: () => ({
    isRunning: false,
    dataList: [],
    attackHistory: [],
    timer: null
  }),

  actions: {
    start() {
      if (this.timer) return
      this.isRunning = true

      this.timer = setInterval(() => {
        const timestamp = this.getBeijingTime()
        const isAttack = Math.random() < 0.015  // 异常率控制约 1.5%

        const sample = {
          time: timestamp,
          src_ip: this.randomIp(),
          src_port: this.randomPort(),
          dst_ip: '10.30.247.240',
          dst_port: 443,
          label: isAttack ? 'PortScan' : 'Benign',
          confidence: isAttack
            ? (Math.random() * 0.15 + 0.8)     // 攻击置信度 0.80–0.95
            : (Math.random() * 0.1 + 0.85)     // 正常置信度 0.85–0.95
        }

        this.dataList.push(sample)

        // 更新图表（最多保留30条）
        if (this.attackHistory.length >= 30) this.attackHistory.shift()
        this.attackHistory.push(isAttack ? 1 : 0)
      }, 1000)
    },

    stop() {
      if (this.timer) {
        clearInterval(this.timer)
        this.timer = null
      }
      this.isRunning = false
    },

    // 获取中国北京时间（UTC+8）
    getBeijingTime() {
      const now = new Date()
      const delaySec = this.rand(6, 10)                    // 模拟监听延迟
      const offsetMs = (8 * 60 * 60 - delaySec) * 1000     // 北京时间减去延迟秒数
      const beijingTime = new Date(now.getTime() + offsetMs)

      return beijingTime.getFullYear() + '-' +
            String(beijingTime.getMonth() + 1).padStart(2, '0') + '-' +
            String(beijingTime.getDate()).padStart(2, '0') + ' ' +
            String(beijingTime.getHours()).padStart(2, '0') + ':' +
            String(beijingTime.getMinutes()).padStart(2, '0') + ':' +
            String(beijingTime.getSeconds()).padStart(2, '0')
    },


    randomIp() {
      return `${this.rand(10, 250)}.${this.rand(0, 255)}.${this.rand(0, 255)}.${this.rand(1, 254)}`
    },

    randomPort() {
      return this.rand(1024, 65535)
    },

    rand(min, max) {
      return Math.floor(Math.random() * (max - min + 1)) + min
    }
  }
})
