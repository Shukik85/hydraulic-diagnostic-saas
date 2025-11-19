<template>
  <Teleport to="body">
    <Transition
      enter-active-class="transition-opacity duration-200"
      enter-from-class="opacity-0"
      enter-to-class="opacity-100"
      leave-active-class="transition-opacity duration-200"
      leave-from-class="opacity-100"
      leave-to-class="opacity-0"
    >
      <div v-if="props.modelValue" class="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm" @click="close">
        <Transition
          enter-active-class="transition-all duration-200"
          enter-from-class="opacity-0 scale-95"
          enter-to-class="opacity-100 scale-100"
          leave-active-class="transition-all duration-200"
          leave-from-class="opacity-100 scale-100"
          leave-to-class="opacity-0 scale-95"
        >
          <div
            v-if="props.modelValue"
            class="fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 card-metal p-6 w-full max-w-lg"
            @click.stop
          >
            <button 
              class="absolute top-4 right-4 text-text-secondary hover:text-text-primary transition-colors rounded-md p-1 hover:bg-background-hover" 
              @click="close"
              aria-label="Close dialog"
            >
              <Icon name="lucide:x" class="w-5 h-5" />
              <span class="sr-only">Close</span>
            </button>
            <slot />
          </div>
        </Transition>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
interface Props {
  modelValue: boolean
}
const props = defineProps<Props>()
const emit = defineEmits<{ 'update:modelValue': [value: boolean] }>()
const close = () => { emit('update:modelValue', false) }
</script>