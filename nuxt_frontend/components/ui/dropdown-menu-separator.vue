<template>
  <hr
    :class="cn('border-border -mx-1 my-1', className)"
    v-bind="$attrs"
  />
</template>

<script setup lang="ts">
import { cn } from "./utils";

interface Props {
  className?: string;
}

withDefaults(defineProps<Props>(), {});
</script>
