<template>
  <div class="relative">
    <button
      ref="triggerRef"
      :class="cn('flex items-center gap-2', triggerClass)"
      @click="toggle"
      v-bind="$attrs"
    >
      <slot name="trigger" />
    </button>
    <Transition
      enter-active-class="transition-all duration-200"
      enter-from-class="opacity-0 scale-95"
      enter-to-class="opacity-100 scale-100"
      leave-active-class="transition-all duration-200"
      leave-from-class="opacity-100 scale-100"
      leave-to-class="opacity-0 scale-95"
    >
      <div
        v-if="isOpen"
        ref="contentRef"
        :class="
          cn(
            'absolute z-50 bg-popover text-popover-foreground rounded-md border shadow-md p-1 min-w-32',
            contentClass
          )
        "
        :style="{ top: position.top + 'px', left: position.left + 'px' }"
      >
        <slot />
      </div>
    </Transition>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue';
import { cn } from './utils';

interface Props {
  triggerClass?: string;
  contentClass?: string;
}

const props = withDefaults(defineProps<Props>(), {});

const isOpen = ref(false);
const triggerRef = ref<HTMLElement>();
const contentRef = ref<HTMLElement>();
const position = ref({ top: 0, left: 0 });

const toggle = () => {
  isOpen.value = !isOpen.value;
  if (isOpen.value) {
    updatePosition();
  }
};

const updatePosition = () => {
  if (triggerRef.value) {
    const rect = triggerRef.value.getBoundingClientRect();
    position.value = {
      top: rect.bottom,
      left: rect.left,
    };
  }
};

const close = () => {
  isOpen.value = false;
};

const handleClickOutside = (event: Event) => {
  if (
    triggerRef.value &&
    !triggerRef.value.contains(event.target as Node) &&
    contentRef.value &&
    !contentRef.value.contains(event.target as Node)
  ) {
    close();
  }
};

onMounted(() => {
  document.addEventListener('click', handleClickOutside);
});

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside);
});
</script>
