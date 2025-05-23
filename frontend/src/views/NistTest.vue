<template>
  <div class="container mt-4">
    <h2 class="mb-3 text-center">NIST 随机性测试结果</h2>
    <table class="table table-bordered">
      <thead>
        <tr>
          <th>测试名称</th>
          <th>P 值</th>
          <th>结果</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(result, index) in testResults" :key="index">
          <td>{{ result.test_name }}</td>
          <td>{{ result.p_value }}</td>
          <td>
            <span :class="{'text-success': result.p_value >= 0.01, 'text-danger': result.p_value < 0.01}">
              {{ result.p_value >= 0.01 ? '通过' : '未通过' }}
            </span>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script>
export default {
  data() {
    return {
      testResults: []
    };
  },
  mounted() {
    // 示例：从后端获取测试结果
    fetch('http://localhost:5000/api/nist-test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ binary_data: '101010101...' })
    })

      .then(response => response.json())
      .then(data => {
        this.testResults = data.results;
      });
  }
};
</script>

<style scoped>
.text-success {
  color: green;
}
.text-danger {
  color: red;
}
</style>
