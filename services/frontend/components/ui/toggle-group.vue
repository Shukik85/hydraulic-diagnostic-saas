<template>
  <div :class="[
      'group/toggle-group flex w-fit items-center rounded-lg shadow border border-steel-medium bg-steel-darker',
      'p-1 gap-1',
      className
    ]"
    v-bind="$attrs"
  >
    <slot />
  </div>
</template>
<script setup lang="ts">// ...Остальной код без изменений...</script>