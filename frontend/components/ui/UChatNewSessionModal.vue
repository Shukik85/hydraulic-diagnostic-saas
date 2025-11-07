<template>
  <UModal
    :model-value="modelValue"
    @update:model-value="$emit('update:modelValue', $event)"
    :title="$t('chat.newSession.title')"
    :description="$t('chat.subtitle')"
    size="md"
    :loading="loading"
  >
    <form @submit.prevent="onSubmit" class="space-y-4">
      <div>
        <label class="u-label" for="chat-session-title">
          {{ $t('ui.name') }}
        </label>
        <input
          id="chat-session-title"
          v-model.trim="title"
          type="text"
          class="u-input"
          :placeholder="$t('chat.newSession.title') + '...'
          "
          :disabled="loading"
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
        :disabled="loading"
      >
        {{ $t('ui.cancel') }}
      </button>
      <button
        type="button"
        class="u-btn u-btn-primary flex-1"
        @click="onSubmit"
        :disabled="loading || !title"
      >
        <Icon name="heroicons:plus" class="w-4 h-4 mr-2" />
        {{ $t('ui.create') }}
      </button>
    </template>
  </UModal>
</template>

<script setup lang="ts">
const props = withDefaults(defineProps<{ 
  modelValue: boolean
  loading?: boolean
  initialTitle?: string
}>(), {
  loading: false,
  initialTitle: ''
})

const emit = defineEmits<{ 
  'update:modelValue': [value: boolean]
  'submit': [payload: { title: string }]
  'cancel': []
}>()

const title = ref(props.initialTitle)
const titleInputRef = ref<HTMLInputElement>()

const onSubmit = () => {
  if (!title.value.trim() || props.loading) return
  emit('submit', { title: title.value.trim() })
}

const onCancel = () => {
  if (props.loading) return
  emit('cancel')
  emit('update:modelValue', false)
}

watch(() => props.modelValue, (isOpen) => {
  if (isOpen) {
    nextTick(() => titleInputRef.value?.focus())
  } else {
    // Reset on close
    setTimeout(() => { title.value = '' }, 200)
  }
})
</script>
