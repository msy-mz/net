import { defineStore } from 'pinia'

export const useListenStore = defineStore('listen', {
  state: () => ({
    isRunning: false,
    dataList: [],
    attackHistory: Array(30).fill(0),  // 初始为 30 个 0，确保图表初始化有数据
    timer: null,
    timeout: null
  }),
  actions: {
    start() {
      if (this.isRunning) return
      this.isRunning = true
      this.dataList = []               // 每次监听清空旧数据
      this.attackHistory = Array(30).fill(0)
      this.simulateTraffic()
      this.startChartUpdate()
    },
    stop() {
      this.isRunning = false
      clearInterval(this.timer)
      clearTimeout(this.timeout)
    },
    addData(data) {
      this.dataList.push(data)
      if (this.dataList.length > 50) this.dataList.shift()
    },
    simulateTraffic() {
      if (!this.isRunning) return
      const count = Math.floor(Math.random() * 6) + 3
      for (let i = 0; i < count; i++) {
        const now = new Date()
        this.addData({
          time: now.toLocaleTimeString(),
          timestamp: now.getTime(),
          src_ip: this.randomPublicIP(),
          dst_ip: '192.168.3.164',
          src_port: this.randomPort(),
          dst_port: 8888,
          label: this.generateLabel(),
          confidence: parseFloat((Math.random() * 0.3 + 0.7).toFixed(3))
        })
      }
      const delay = Math.floor(Math.random() * 700) + 800
      this.timeout = setTimeout(this.simulateTraffic, delay)
    },
    startChartUpdate() {
      this.timer = setInterval(() => {
        const now = Date.now()
        const oneSecAgo = now - 1000
        // 取最近 3 秒内的攻击数作为趋势近似值（更鲁棒）
        const recentAttacks = this.dataList.filter(
          d => d.label !== 'Benign' && now - d.timestamp <= 3000
        ).length

        this.attackHistory.push(recentAttacks)
        if (this.attackHistory.length > 30) this.attackHistory.shift()
      }, 1000)
    },
    randomPort() {
      return Math.floor(49152 + Math.random() * (65535 - 49152))
    },
    randomPublicIP() {
      return `${Math.floor(1 + Math.random() * 223)}.${Math.floor(Math.random() * 256)}.${Math.floor(Math.random() * 256)}.${Math.floor(Math.random() * 256)}`
    },
    generateLabel() {
      return Math.random() < 0.99 ? 'Benign' : ['Botnet', 'DoS', 'BruteForce', 'Infiltration'][Math.floor(Math.random() * 4)]
    }
  }
})
