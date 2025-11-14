<template>
  <nav aria-label="breadcrumb" class="flex">
    <ol class="flex items-center space-x-1">
      <li v-for="(item, index) in items" :key="item.href || index" class="flex items-center">
        <NuxtLink
          v-if="item.href && index < items.length - 1"
          :to="item.href"
          class="text-sm text-text-secondary hover:text-primary-400 transition-colors duration-200 font-medium"
        >
          {{ item.label }}
        </NuxtLink>
        <span
          v-else
          :class="[
            'text-sm',
            index === items.length - 1 
              ? 'text-text-primary font-semibold' 
              : 'text-text-secondary',
          ]"
        >
          {{ item.label }}
        </span>
        <Icon
          v-if="index < items.length - 1"
          name="lucide:chevron-right"
          class="h-4 w-4 mx-2 text-steel-light"
        />
      </li>
    </ol>
  </nav>
</template>

<script setup lang="ts">
interface BreadcrumbItem {
  label: string
  href?: string
}

interface Props {
  items: BreadcrumbItem[]
}

defineProps<Props>()
</script>