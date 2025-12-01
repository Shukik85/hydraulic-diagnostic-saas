import { defineStore } from 'pinia';

export interface Toast {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  message: string;
  duration?: number;
}

export interface UiState {
  isSidebarOpen: boolean;
  isMobileSidebarOpen: boolean;
  toasts: Toast[];
  isModalOpen: boolean;
  modalContent: string | null;
  isLoading: boolean;
  theme: 'light' | 'dark' | 'auto';
}

export const useUiStore = defineStore('ui', {
  state: (): UiState => ({
    isSidebarOpen: true,
    isMobileSidebarOpen: false,
    toasts: [],
    isModalOpen: false,
    modalContent: null,
    isLoading: false,
    theme: 'auto',
  }),

  getters: {
    activeToasts: (state) => state.toasts,
    hasActiveToasts: (state) => state.toasts.length > 0,
    currentTheme: (state) => state.theme,
  },

  actions: {
    toggleSidebar() {
      this.isSidebarOpen = !this.isSidebarOpen;
      // Persist to localStorage
      if (import.meta.client) {
        localStorage.setItem('sidebar_open', JSON.stringify(this.isSidebarOpen));
      }
    },

    toggleMobileSidebar() {
      this.isMobileSidebarOpen = !this.isMobileSidebarOpen;
    },

    closeMobileSidebar() {
      this.isMobileSidebarOpen = false;
    },

    showToast(toast: Omit<Toast, 'id'>) {
      const id = `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      const newToast: Toast = {
        ...toast,
        id,
        duration: toast.duration ?? 5000,
      };

      this.toasts.push(newToast);

      // Auto-remove after duration
      if (newToast.duration && newToast.duration > 0) {
        setTimeout(() => {
          this.removeToast(id);
        }, newToast.duration);
      }
    },

    removeToast(id: string) {
      const index = this.toasts.findIndex((t) => t.id === id);
      if (index !== -1) {
        this.toasts.splice(index, 1);
      }
    },

    clearToasts() {
      this.toasts = [];
    },

    openModal(content: string) {
      this.modalContent = content;
      this.isModalOpen = true;
    },

    closeModal() {
      this.isModalOpen = false;
      this.modalContent = null;
    },

    setLoading(loading: boolean) {
      this.isLoading = loading;
    },

    setTheme(theme: 'light' | 'dark' | 'auto') {
      this.theme = theme;
      
      // Apply theme to document
      if (import.meta.client) {
        if (theme === 'auto') {
          const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
          document.documentElement.setAttribute('data-color-scheme', prefersDark ? 'dark' : 'light');
        } else {
          document.documentElement.setAttribute('data-color-scheme', theme);
        }
        
        // Save to localStorage
        localStorage.setItem('theme', theme);
      }
    },

    initTheme() {
      if (import.meta.client) {
        const savedTheme = localStorage.getItem('theme') as 'light' | 'dark' | 'auto' | null;
        if (savedTheme) {
          this.setTheme(savedTheme);
        } else {
          // Default to auto
          this.setTheme('auto');
        }
      }
    },

    /**
     * Restore theme from localStorage (alias for initTheme)
     * Called from app.vue on mount
     */
    restoreTheme() {
      this.initTheme();
    },

    /**
     * Restore sidebar state from localStorage
     * Called from app.vue on mount
     */
    restoreSidebar() {
      if (import.meta.client) {
        const saved = localStorage.getItem('sidebar_open');
        if (saved !== null) {
          try {
            this.isSidebarOpen = JSON.parse(saved);
          } catch (error) {
            console.error('Failed to parse sidebar state:', error);
          }
        }
      }
    },
  },
});
