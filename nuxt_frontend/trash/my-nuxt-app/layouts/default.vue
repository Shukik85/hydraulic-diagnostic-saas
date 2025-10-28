<template>
  <div>
    <header>
      <h1>Мой Nuxt.js Приложение</h1>
    </header>
    <nuxt />
    <footer>
      <p>© 2025 Мое приложение</p>
    </footer>
  </div>
</template>

<script lang="ts">
export default {
  layout: 'default',
};
</script>

<style scoped>
header {
  background-color: #f8f9fa;
  padding: 1rem;
  text-align: center;
}

footer {
  background-color: #f8f9fa;
  padding: 1rem;
  text-align: center;
}
</style>
