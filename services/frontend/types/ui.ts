/**
 * UI State Types
 * Frontend state management
 */

/**
 * Toast notification types
 */
export type ToastType = 'success' | 'error' | 'warning' | 'info';

/**
 * Toast message
 */
export interface ToastMessage {
  id: string;
  type: ToastType;
  title?: string;
  message: string;
  duration?: number;
  dismissible?: boolean;
  createdAt: Date;
}

/**
 * Modal state
 */
export interface ModalState {
  isOpen: boolean;
  title?: string;
  content?: unknown;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  closeOnBackdrop?: boolean;
  showCloseButton?: boolean;
  onConfirm?: () => void | Promise<void>;
  onCancel?: () => void;
}

/**
 * Page state
 */
export interface PageState<T = unknown> {
  loading: boolean;
  error: Error | null;
  data: T | null;
}

/**
 * Table state
 */
export interface TableState<T = unknown> {
  data: T[];
  loading: boolean;
  error: Error | null;
  pagination: {
    page: number;
    limit: number;
    total: number;
  };
  sorting?: {
    column: string;
    direction: 'asc' | 'desc';
  };
  filters?: Record<string, unknown>;
}

/**
 * Form state
 */
export interface FormState<T = Record<string, unknown>> {
  values: T;
  errors: Partial<Record<keyof T, string>>;
  touched: Partial<Record<keyof T, boolean>>;
  isSubmitting: boolean;
  isValid: boolean;
}

/**
 * Theme
 */
export type Theme = 'light' | 'dark' | 'system';

/**
 * Sidebar state
 */
export interface SidebarState {
  isOpen: boolean;
  isPinned: boolean;
  activeSection?: string;
}

/**
 * Breadcrumb item
 */
export interface BreadcrumbItem {
  label: string;
  to?: string;
  icon?: string;
}
