/**
 * UI state types
 */

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
}

/**
 * Toast type
 */
export type ToastType = 'success' | 'error' | 'warning' | 'info';

/**
 * Modal state
 */
export interface ModalState {
  isOpen: boolean;
  title?: string;
  content?: string;
  component?: string;
  props?: Record<string, unknown>;
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
 * Theme
 */
export type Theme = 'light' | 'dark' | 'system';

/**
 * Breadcrumb item
 */
export interface BreadcrumbItem {
  label: string;
  to?: string;
  active?: boolean;
}

/**
 * Table column
 */
export interface TableColumn<T = unknown> {
  key: keyof T | string;
  label: string;
  sortable?: boolean;
  filterable?: boolean;
  width?: string;
  align?: 'left' | 'center' | 'right';
  formatter?: (value: unknown, row: T) => string;
}

/**
 * Sort direction
 */
export type SortDirection = 'asc' | 'desc';

/**
 * Filter
 */
export interface Filter {
  field: string;
  operator: FilterOperator;
  value: unknown;
}

/**
 * Filter operator
 */
export type FilterOperator = 'eq' | 'ne' | 'gt' | 'gte' | 'lt' | 'lte' | 'like' | 'in';
