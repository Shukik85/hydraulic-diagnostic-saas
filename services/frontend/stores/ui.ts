/**
 * UI state store
 */

import { defineStore } from 'pinia';
import type { ToastMessage, ModalState, Theme } from '~/types';

export const useUiStore = defineStore('ui', () => {
  // State
  const toasts = ref<ToastMessage[]>([]);
  const modals = ref<Record<string, ModalState>>({});
  const theme = ref<Theme>('system');
  const sidebarOpen = ref(true);

  // Getters
  const activeToasts = computed(() => toasts.value);
  const activeModals = computed(() =>
    Object.entries(modals.value)
      .filter(([, modal]) => modal.isOpen)
      .map(([name]) => name)
  );
  const hasActiveModal = computed(() => activeModals.value.length > 0);

  // Toast Actions
  const addToast = (toast: ToastMessage): void => {
    toasts.value.push(toast);
  };

  const removeToast = (id: string): void => {
    const index = toasts.value.findIndex((t) => t.id === id);
    if (index !== -1) {
      toasts.value.splice(index, 1);
    }
  };

  const clearToasts = (): void => {
    toasts.value = [];
  };

  // Modal Actions
  const openModal = (name: string, options: Partial<ModalState> = {}): void => {
    modals.value[name] = {
      isOpen: true,
      ...options,
    };
  };

  const closeModal = (name: string): void => {
    if (modals.value[name]) {
      modals.value[name].isOpen = false;
    }
  };

  const closeAllModals = (): void => {
    Object.keys(modals.value).forEach((name) => {
      modals.value[name].isOpen = false;
    });
  };

  // Theme Actions
  const setTheme = (newTheme: Theme): void => {
    theme.value = newTheme;

    // Persist to localStorage
    if (process.client) {
      localStorage.setItem('theme', newTheme);
      applyTheme(newTheme);
    }
  };

  const toggleTheme = (): void => {
    const newTheme = theme.value === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
  };

  const applyTheme = (themeValue: Theme): void => {
    if (!process.client) {
      return;
    }

    const root = document.documentElement;

    if (themeValue === 'system') {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      root.classList.toggle('dark', prefersDark);
    } else {
      root.classList.toggle('dark', themeValue === 'dark');
    }
  };

  const restoreTheme = (): void => {
    if (process.client) {
      const savedTheme = localStorage.getItem('theme') as Theme | null;
      if (savedTheme) {
        theme.value = savedTheme;
        applyTheme(savedTheme);
      } else {
        applyTheme('system');
      }
    }
  };

  // Sidebar Actions
  const toggleSidebar = (): void => {
    sidebarOpen.value = !sidebarOpen.value;

    // Persist to localStorage
    if (process.client) {
      localStorage.setItem('sidebarOpen', JSON.stringify(sidebarOpen.value));
    }
  };

  const restoreSidebar = (): void => {
    if (process.client) {
      const saved = localStorage.getItem('sidebarOpen');
      if (saved !== null) {
        sidebarOpen.value = JSON.parse(saved);
      }
    }
  };

  return {
    // State
    toasts,
    modals,
    theme,
    sidebarOpen,

    // Getters
    activeToasts,
    activeModals,
    hasActiveModal,

    // Toast Actions
    addToast,
    removeToast,
    clearToasts,

    // Modal Actions
    openModal,
    closeModal,
    closeAllModals,

    // Theme Actions
    setTheme,
    toggleTheme,
    applyTheme,
    restoreTheme,

    // Sidebar Actions
    toggleSidebar,
    restoreSidebar,
  };
});
