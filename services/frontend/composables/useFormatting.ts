/**
 * Formatting utility composable
 */

export interface UseFormattingReturn {
  formatCurrency: (value: number, currency?: string, locale?: string) => string;
  formatDate: (date: Date | string, format?: string, locale?: string) => string;
  formatDuration: (milliseconds: number) => string;
  formatBytes: (bytes: number, decimals?: number) => string;
  formatPercent: (value: number, decimals?: number) => string;
  formatNumber: (value: number, decimals?: number, locale?: string) => string;
  truncate: (text: string, length: number, suffix?: string) => string;
}

/**
 * Formatting utility composable
 */
export const useFormatting = (): UseFormattingReturn => {
  const { locale } = useI18n();

  /**
   * Format currency value
   */
  const formatCurrency = (
    value: number,
    currency = 'USD',
    localeStr = locale.value
  ): string => {
    return new Intl.NumberFormat(localeStr, {
      style: 'currency',
      currency,
    }).format(value);
  };

  /**
   * Format date
   */
  const formatDate = (
    date: Date | string,
    format = 'short',
    localeStr = locale.value
  ): string => {
    const dateObj = typeof date === 'string' ? new Date(date) : date;

    const formats: Record<string, Intl.DateTimeFormatOptions> = {
      short: { year: 'numeric', month: 'short', day: 'numeric' },
      long: {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      },
      time: { hour: '2-digit', minute: '2-digit', second: '2-digit' },
      date: { year: 'numeric', month: '2-digit', day: '2-digit' },
    };

    return new Intl.DateTimeFormat(localeStr, formats[format] || formats.short).format(dateObj);
  };

  /**
   * Format duration (milliseconds to human-readable)
   */
  const formatDuration = (milliseconds: number): string => {
    const seconds = Math.floor(milliseconds / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) {
      return `${days}d ${hours % 24}h`;
    }
    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    }
    if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    }
    return `${seconds}s`;
  };

  /**
   * Format bytes to human-readable size
   */
  const formatBytes = (bytes: number, decimals = 2): string => {
    if (bytes === 0) {
      return '0 Bytes';
    }

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(decimals))} ${sizes[i]}`;
  };

  /**
   * Format percentage
   */
  const formatPercent = (value: number, decimals = 2): string => {
    return `${(value * 100).toFixed(decimals)}%`;
  };

  /**
   * Format number with locale
   */
  const formatNumber = (
    value: number,
    decimals = 0,
    localeStr = locale.value
  ): string => {
    return new Intl.NumberFormat(localeStr, {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    }).format(value);
  };

  /**
   * Truncate text
   */
  const truncate = (text: string, length: number, suffix = '...'): string => {
    if (text.length <= length) {
      return text;
    }
    return text.substring(0, length - suffix.length) + suffix;
  };

  return {
    formatCurrency,
    formatDate,
    formatDuration,
    formatBytes,
    formatPercent,
    formatNumber,
    truncate,
  };
};
