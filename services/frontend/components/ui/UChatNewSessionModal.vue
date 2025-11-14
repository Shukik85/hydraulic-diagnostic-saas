<template>
  <UModal :model-value="modelValue" @update:model-value="$emit('update:modelValue', $event)" :title="$t('chat.newSession.title')" :description="$t('chat.subtitle')" size="md" :loading="loading">
    <form @submit.prevent="onSubmit" class="space-y-4">
      <div>
        <label class="u-label" for="chat-session-title">{{ $t('ui.name') }}</label>
        <input id="chat-session-title" v-model.trim="title" type="text" class="u-input" :placeholder="$t('chat.newSession.title') + '...'" :disabled="loading" maxlength="200" ref="titleInputRef" required />
      </div>
    </form>
    <template #footer>
      <button type="button" class="u-btn u-btn-secondary flex-1" @click="onCancel" :disabled="loading">{{ $t('ui.cancel') }}</button>
      <button type="button" class="u-btn u-btn-primary flex-1" @click="onSubmit" :disabled="loading || !title">
        <Icon name="heroicons:plus" class="w-4 h-4 mr-2" />{{ $t('ui.create') }}
      </button>
    </template>
  </UModal>
</template>
<script setup lang="ts">// ...без изменений...</script>