<script setup lang="ts">
interface Props {
  modelValue: boolean
  loading?: boolean
  title?: string
}

interface Emits {
  (e: 'update:modelValue', value: boolean): void
  (e: 'submit', data: { title: string }): void
  (e: 'cancel'): void
}

const props = withDefaults(defineProps<Props>(), {
  loading: false
})

const emit = defineEmits<Emits>()

const { t } = useI18n()
const title = ref('')
const titleInputRef = ref<HTMLInputElement | null>(null)

const onSubmit = () => {
  if (title.value.trim()) {
    emit('submit', { title: title.value.trim() })
    title.value = ''
  }
}

const onCancel = () => {
  emit('cancel')
  emit('update:modelValue', false)
  title.value = ''
}

watch(() => props.modelValue, (newVal) => {
  if (newVal && titleInputRef.value) {
    nextTick(() => {
      titleInputRef.value?.focus()
    })
  }
})
</script>

<template>
  <UModal 
    :model-value="props.modelValue" 
    @update:model-value="emit('update:modelValue', $event)" 
    :title="t('chat.newSession.title')" 
    :description="t('chat.subtitle')" 
    size="md" 
    :loading="props.loading"
  >
    <form @submit.prevent="onSubmit" class="space-y-4">
      <div>
        <label class="u-label" for="chat-session-title">{{ t('ui.name') }}</label>
        <input 
          id="chat-session-title" 
          v-model.trim="title" 
          type="text" 
          class="u-input" 
          :placeholder="t('chat.newSession.title') + '...'" 
          :disabled="props.loading" 
          maxlength="200" 
          ref="titleInputRef" 
          required 
        />
      </div>
    </form>
    
    <template #footer>
      <button 
        type="button" 
        class="u-btn u-btn-secondary flex-1" 
        @click="onCancel" 
        :disabled="props.loading"
      >
        {{ t('ui.cancel') }}
      </button>
      <button 
        type="button" 
        class="u-btn u-btn-primary flex-1" 
        @click="onSubmit" 
        :disabled="props.loading || !title"
      >
        <Icon name="heroicons:plus" class="w-4 h-4 mr-2" />{{ t('ui.create') }}
      </button>
    </template>
  </UModal>
</template>
