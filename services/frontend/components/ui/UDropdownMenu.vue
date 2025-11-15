<template>
  <DropdownMenuRoot v-model:open="isOpen">
    <DropdownMenuTrigger as-child>
      <slot name="trigger" />
    </DropdownMenuTrigger>
    <DropdownMenuPortal>
      <DropdownMenuContent
        :class="
          cn(
            'z-50 min-w-[12rem] overflow-hidden',
            'rounded-lg border border-steel-medium',
            'bg-steel-darker shadow-lg',
            'p-1',
            'animate-in fade-in-80 zoom-in-95',
            'data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95',
            props.contentClass
          )
        "
        :side-offset="4"
        v-bind="$attrs"
      >
        <slot />
      </DropdownMenuContent>
    </DropdownMenuPortal>
  </DropdownMenuRoot>
</template>
<script setup lang="ts">
import {
  DropdownMenuContent,
  DropdownMenuPortal,
  DropdownMenuRoot,
  DropdownMenuTrigger,
} from 'radix-vue'
import { ref } from 'vue'
import { cn } from './utils'
interface Props {
  contentClass?: string
}
const props = withDefaults(defineProps<Props>(), {})
const isOpen = ref(false)
</script>