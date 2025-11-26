/**
 * Formatter Utilities
 * Pure formatting functions (no composable dependencies)
 */

import type { SensorType, AnomalySeverity } from '~/types';

// =============================================================================
// Number Formatting
// =============================================================================

/**
 * Format number with localization
 */
export function formatNumber(
  value: number,
  options: Intl.NumberFormatOptions = {},
  locale: string = 'en-US'
): string {
  return new Intl.NumberFormat(locale, options).format(value);
}

/**
 * Format number with suffix (K, M, B)
 */
export function formatNumberCompact(value: number, locale: string = 'en-US'): string {
  return new Intl.NumberFormat(locale, {
    notation: 'compact',
    compactDisplay: 'short',
  }).format(value);
}

// =============================================================================
// Currency Formatting
// =============================================================================

/**
 * Format currency
 */
export function formatCurrency(
  value: number,
  currency: string = 'USD',
  locale: string = 'en-US'
): string {
  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency,
  }).format(value);
}

/**
 * Format currency compact (e.g., $1.5K, $2.3M)
 */
export function formatCurrencyCompact(
  value: number,
  currency: string = 'USD',
  locale: string = 'en-US'
): string {
  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency,
    notation: 'compact',
    compactDisplay: 'short',
  }).format(value);
}

// =============================================================================
// Date Formatting
// =============================================================================

/**
 * Format date
 */
export function formatDate(
  date: Date | string | number,
  format: 'short' | 'long' | 'time' | 'datetime' | 'iso' = 'short',
  locale: string = 'en-US'
): string {
  const d = typeof date === 'string' || typeof date === 'number' ? new Date(date) : date;

  switch (format) {
    case 'short':
      return d.toLocaleDateString(locale);
    case 'long':
      return d.toLocaleDateString(locale, {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
      });
    case 'time':
      return d.toLocaleTimeString(locale);
    case 'datetime':
      return `${d.toLocaleDateString(locale)} ${d.toLocaleTimeString(locale)}`;
    case 'iso':
      return d.toISOString();
    default:
      return d.toLocaleDateString(locale);
  }
}

/**
 * Format relative time (e.g., "2 hours ago")
 */
export function formatRelativeTime(date: Date | string | number, locale: string = 'en-US'): string {
  const d = typeof date === 'string' || typeof date === 'number' ? new Date(date) : date;
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffSeconds = Math.floor(diffMs / 1000);
  const diffMinutes = Math.floor(diffSeconds / 60);
  const diffHours = Math.floor(diffMinutes / 60);
  const diffDays = Math.floor(diffHours / 24);
  const diffWeeks = Math.floor(diffDays / 7);
  const diffMonths = Math.floor(diffDays / 30);
  const diffYears = Math.floor(diffDays / 365);

  if (diffYears > 0) return `${diffYears} year${diffYears > 1 ? 's' : ''} ago`;
  if (diffMonths > 0) return `${diffMonths} month${diffMonths > 1 ? 's' : ''} ago`;
  if (diffWeeks > 0) return `${diffWeeks} week${diffWeeks > 1 ? 's' : ''} ago`;
  if (diffDays > 0) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
  if (diffHours > 0) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
  if (diffMinutes > 0) return `${diffMinutes} minute${diffMinutes > 1 ? 's' : ''} ago`;
  if (diffSeconds > 0) return `${diffSeconds} second${diffSeconds > 1 ? 's' : ''} ago`;
  return 'just now';
}

// =============================================================================
// Duration Formatting
// =============================================================================

/**
 * Format duration in milliseconds
 */
export function formatDuration(ms: number, verbose: boolean = false): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (verbose) {
    const parts: string[] = [];
    if (days > 0) parts.push(`${days} day${days > 1 ? 's' : ''}`);
    if (hours % 24 > 0) parts.push(`${hours % 24} hour${hours % 24 > 1 ? 's' : ''}`);
    if (minutes % 60 > 0) parts.push(`${minutes % 60} minute${minutes % 60 > 1 ? 's' : ''}`);
    if (seconds % 60 > 0) parts.push(`${seconds % 60} second${seconds % 60 > 1 ? 's' : ''}`);
    return parts.join(', ');
  }

  if (days > 0) return `${days}d ${hours % 24}h`;
  if (hours > 0) return `${hours}h ${minutes % 60}m`;
  if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
  return `${seconds}s`;
}

// =============================================================================
// Bytes Formatting
// =============================================================================

/**
 * Format bytes to human-readable string
 */
export function formatBytes(bytes: number, decimals: number = 2): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(decimals))} ${sizes[i]}`;
}

// =============================================================================
// Percentage Formatting
// =============================================================================

/**
 * Format percentage
 */
export function formatPercent(
  value: number,
  decimals: number = 1,
  locale: string = 'en-US'
): string {
  return new Intl.NumberFormat(locale, {
    style: 'percent',
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value / 100);
}

/**
 * Format percentage with sign
 */
export function formatPercentChange(
  value: number,
  decimals: number = 1,
  locale: string = 'en-US'
): string {
  const sign = value >= 0 ? '+' : '';
  return `${sign}${formatPercent(value, decimals, locale)}`;
}

// =============================================================================
// Sensor Value Formatting
// =============================================================================

/**
 * Format pressure
 */
export function formatPressure(value: number, unit: string = 'bar'): string {
  return `${value.toFixed(1)} ${unit}`;
}

/**
 * Format temperature
 */
export function formatTemperature(value: number, unit: string = '°C'): string {
  return `${value.toFixed(1)}${unit}`;
}

/**
 * Format flow rate
 */
export function formatFlow(value: number, unit: string = 'L/min'): string {
  return `${value.toFixed(1)} ${unit}`;
}

/**
 * Format vibration
 */
export function formatVibration(value: number, unit: string = 'mm/s'): string {
  return `${value.toFixed(2)} ${unit}`;
}

/**
 * Format sensor value based on type
 */
export function formatSensorValue(value: number, type: SensorType, unit?: string): string {
  switch (type) {
    case 'pressure':
      return formatPressure(value, unit);
    case 'temperature':
      return formatTemperature(value, unit);
    case 'flow':
      return formatFlow(value, unit);
    case 'vibration':
      return formatVibration(value, unit);
    case 'position':
      return `${value.toFixed(1)} ${unit || 'mm'}`;
    default:
      return value.toString();
  }
}

// =============================================================================
// Status Formatting
// =============================================================================

/**
 * Format anomaly severity
 */
export function formatSeverity(severity: AnomalySeverity): string {
  const severityMap: Record<AnomalySeverity, string> = {
    low: 'Low',
    medium: 'Medium',
    high: 'High',
    critical: 'Critical',
  };
  return severityMap[severity];
}

/**
 * Format sensor status
 */
export function formatSensorStatus(
  status: 'online' | 'offline' | 'error' | 'calibrating'
): string {
  const statusMap = {
    online: 'Online',
    offline: 'Offline',
    error: 'Error',
    calibrating: 'Calibrating',
  };
  return statusMap[status];
}

// =============================================================================
// String Formatting
// =============================================================================

/**
 * Truncate string with ellipsis
 */
export function truncate(str: string, maxLength: number): string {
  if (str.length <= maxLength) return str;
  return `${str.substring(0, maxLength - 3)}...`;
}

/**
 * Capitalize first letter
 */
export function capitalize(str: string): string {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * Convert to title case
 */
export function toTitleCase(str: string): string {
  return str
    .toLowerCase()
    .split(' ')
    .map((word) => capitalize(word))
    .join(' ');
}

/**
 * Convert snake_case to Title Case
 */
export function snakeToTitle(str: string): string {
  return str
    .split('_')
    .map((word) => capitalize(word))
    .join(' ');
}

/**
 * Convert camelCase to Title Case
 */
export function camelToTitle(str: string): string {
  return str
    .replace(/([A-Z])/g, ' $1')
    .trim()
    .split(' ')
    .map((word) => capitalize(word))
    .join(' ');
}
