<template>
  <div class="skeleton-card" :class="sizeClass">
    <div v-if="showHeader" class="skeleton-header">
      <div class="skeleton-avatar"></div>
      <div class="skeleton-text-block">
        <div class="skeleton-line skeleton-line-title"></div>
        <div class="skeleton-line skeleton-line-subtitle"></div>
      </div>
    </div>
    
    <div class="skeleton-body">
      <div v-for="i in lines" :key="i" class="skeleton-line" :style="getLineWidth(i)"></div>
    </div>
    
    <div v-if="showFooter" class="skeleton-footer">
      <div class="skeleton-button"></div>
      <div class="skeleton-button skeleton-button-secondary"></div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface Props {
  size?: 'sm' | 'md' | 'lg'
  lines?: number
  showHeader?: boolean
  showFooter?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  size: 'md',
  lines: 3,
  showHeader: true,
  showFooter: true
})

const sizeClass = computed(() => `skeleton-${props.size}`)

const getLineWidth = (index: number) => {
  const widths = ['100%', '95%', '85%', '90%', '100%']
  return { width: widths[(index - 1) % widths.length] }
}
</script>

<style scoped>
.skeleton-card {
  background: linear-gradient(120deg, #2b3340 0%, #232731 81%);
  border: 2px solid #424c5b;
  border-radius: 0.75rem;
  padding: 1.5rem;
}

.skeleton-sm { padding: 1rem; }
.skeleton-lg { padding: 2rem; }

.skeleton-header {
  display: flex;
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.skeleton-avatar {
  width: 3rem;
  height: 3rem;
  background: linear-gradient(90deg, #232b36 0%, #2b3340 50%, #232b36 100%);
  background-size: 200% 100%;
  border-radius: 0.5rem;
  animation: shimmer 1.5s infinite;
}

.skeleton-text-block {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.skeleton-body {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  margin-bottom: 1.5rem;
}

.skeleton-line {
  height: 0.875rem;
  background: linear-gradient(90deg, #232b36 0%, #2b3340 50%, #232b36 100%);
  background-size: 200% 100%;
  border-radius: 0.25rem;
  animation: shimmer 1.5s infinite;
}

.skeleton-line-title {
  height: 1rem;
  width: 60%;
}

.skeleton-line-subtitle {
  height: 0.75rem;
  width: 40%;
}

.skeleton-footer {
  display: flex;
  gap: 1rem;
  justify-content: flex-end;
}

.skeleton-button {
  width: 5rem;
  height: 2.5rem;
  background: linear-gradient(90deg, #232b36 0%, #2b3340 50%, #232b36 100%);
  background-size: 200% 100%;
  border-radius: 0.5rem;
  animation: shimmer 1.5s infinite;
}

.skeleton-button-secondary {
  width: 6rem;
}

@keyframes shimmer {
  0% { background-position: 100% 0; }
  100% { background-position: -100% 0; }
}
</style>