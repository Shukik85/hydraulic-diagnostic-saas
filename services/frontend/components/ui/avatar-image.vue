<template>
  <img :class="cn('aspect-square size-full', className)" v-bind="$attrs" />
</template>

<script setup lang="ts">
import { cn } from './utils';

interface Props {
  className?: string;
}

withDefaults(defineProps<Props>(), {});
</script>
