<template>
  <nav aria-label="breadcrumb" class="flex">
    <ol class="flex items-center space-x-1">
      <li v-for="(item, index) in items" :key="item.href || index" class="flex items-center">
        <NuxtLink
          v-if="item.href && index < items.length - 1"
          :to="item.href"
          class="text-muted-foreground hover:text-foreground transition-colors"
        >
          {{ item.label }}
        </NuxtLink>
        <span
          v-else
          :class="[
            'text-sm',
            index === items.length - 1 ? 'text-foreground font-medium' : 'text-muted-foreground'
          ]"
        >
          {{ item.label }}
        </span>
        <Icon
          v-if="index < items.length - 1"
          name="lucide:chevron-right"
          class="h-4 w-4 mx-1 text-muted-foreground"
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
