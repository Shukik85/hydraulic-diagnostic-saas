/**
 * UI Store
 * Application-wide UI state management
 */

import { defineStore } from 'pinia';
import type { ToastMessage, ModalState, Theme, SidebarState } from '~/types';

interface UIState {
  toasts: ToastMessage[];
  modals: Record<string, ModalState>;
  theme: Theme;
  sidebar: SidebarState;
}

let toastIdCounter = 0;

export const useUIStore = defineStore('ui', {
  state: (): UIState => ({
    toasts: [],
    modals: {},
    theme: 'system',
    sidebar: {
      isOpen: true,
      isPinned: true,
      activeSection: undefined,
    },
  }),

  getters: {
    /**
     * Get active toasts
     */
    activeToasts: (state): ToastMessage[] => {
      return state.toasts;
    },

    /**
     * Check if modal is open
     */
    isModalOpen:
      (state) =>
      (name: string): boolean => {
        return state.modals[name]?.isOpen || false;
      },

    /**
     * Get effective theme (resolve 'system' to actual theme)
     */
    effectiveTheme: (state): 'light' | 'dark' => {
      if (state.theme === 'system') {
        // Check system preference
        if (typeof window !== 'undefined') {
          return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        }
        return 'light';
      }
      return state.theme;
    },
  },

  actions: {
    /**
     * Add toast notification
     */
    addToast(
      message: string,
      type: ToastMessage['type'] = 'info',
      title?: string,
      duration: number = 5000
    ): string {
      const id = `toast-${Date.now()}-${toastIdCounter++}`;

      const toast: ToastMessage = {
        id,
        type,
        title,
        message,
        duration,
        dismissible: true,
        createdAt: new Date(),
      };

      this.toasts.push(toast);

      // Auto-remove after duration
      if (duration > 0) {
        setTimeout(() => {
          this.removeToast(id);
        }, duration);
      }

      return id;
    },

    /**
     * Remove toast by ID
     */
    removeToast(id: string): void {
      const index = this.toasts.findIndex((t) => t.id === id);
      if (index !== -1) {
        this.toasts.splice(index, 1);
      }
    },

    /**
     * Clear all toasts
     */
    clearToasts(): void {
      this.toasts = [];
    },

    /**
     * Show success toast
     */
    showSuccess(message: string, title: string = 'Success'): string {
      return this.addToast(message, 'success', title);
    },

    /**
     * Show error toast
     */
    showError(message: string, title: string = 'Error'): string {
      return this.addToast(message, 'error', title, 0); // Don't auto-dismiss errors
    },

    /**
     * Show warning toast
     */
    showWarning(message: string, title: string = 'Warning'): string {
      return this.addToast(message, 'warning', title);
    },

    /**
     * Show info toast
     */
    showInfo(message: string, title: string = 'Info'): string {
      return this.addToast(message, 'info', title);
    },

    /**
     * Open modal
     */
    openModal(
      name: string,
      options: Partial<ModalState> = {}
    ): void {
      this.modals[name] = {
        isOpen: true,
        title: options.title,
        content: options.content,
        size: options.size || 'md',
        closeOnBackdrop: options.closeOnBackdrop ?? true,
        showCloseButton: options.showCloseButton ?? true,
        onConfirm: options.onConfirm,
        onCancel: options.onCancel,
      };
    },

    /**
     * Close modal
     */
    closeModal(name: string): void {
      if (this.modals[name]) {
        this.modals[name].isOpen = false;
        // Clean up after animation
        setTimeout(() => {
          delete this.modals[name];
        }, 300);
      }
    },

    /**
     * Close all modals
     */
    closeAllModals(): void {
      Object.keys(this.modals).forEach((name) => {
        this.closeModal(name);
      });
    },

    /**
     * Set theme
     */
    setTheme(theme: Theme): void {
      this.theme = theme;
      
      // Persist to localStorage
      if (typeof window !== 'undefined') {
        localStorage.setItem('theme', theme);
        
        // Apply theme to document
        const effectiveTheme = theme === 'system'
          ? window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
          : theme;
        
        document.documentElement.classList.toggle('dark', effectiveTheme === 'dark');
      }
    },

    /**
     * Toggle theme
     */
    toggleTheme(): void {
      const current = this.effectiveTheme;
      this.setTheme(current === 'dark' ? 'light' : 'dark');
    },

    /**
     * Open sidebar
     */
    openSidebar(): void {
      this.sidebar.isOpen = true;
    },

    /**
     * Close sidebar
     */
    closeSidebar(): void {
      this.sidebar.isOpen = false;
    },

    /**
     * Toggle sidebar
     */
    toggleSidebar(): void {
      this.sidebar.isOpen = !this.sidebar.isOpen;
    },

    /**
     * Pin sidebar
     */
    pinSidebar(): void {
      this.sidebar.isPinned = true;
      if (typeof window !== 'undefined') {
        localStorage.setItem('sidebar_pinned', 'true');
      }
    },

    /**
     * Unpin sidebar
     */
    unpinSidebar(): void {
      this.sidebar.isPinned = false;
      if (typeof window !== 'undefined') {
        localStorage.setItem('sidebar_pinned', 'false');
      }
    },

    /**
     * Set active sidebar section
     */
    setActiveSidebarSection(section?: string): void {
      this.sidebar.activeSection = section;
    },

    /**
     * Initialize UI state from localStorage
     */
    init(): void {
      if (typeof window === 'undefined') return;

      // Load theme
      const savedTheme = localStorage.getItem('theme') as Theme | null;
      if (savedTheme) {
        this.setTheme(savedTheme);
      } else {
        this.setTheme('system');
      }

      // Load sidebar state
      const sidebarPinned = localStorage.getItem('sidebar_pinned');
      if (sidebarPinned !== null) {
        this.sidebar.isPinned = sidebarPinned === 'true';
      }

      // Listen for system theme changes
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      mediaQuery.addEventListener('change', (e) => {
        if (this.theme === 'system') {
          document.documentElement.classList.toggle('dark', e.matches);
        }
      });
    },
  },
});
