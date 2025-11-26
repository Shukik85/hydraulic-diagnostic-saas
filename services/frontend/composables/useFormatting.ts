/**
 * Formatting Composable
 * Domain-specific formatters for hydraulic diagnostics
 */

import type { SensorType, AnomalySeverity } from '~/types';

/**
 * Format pressure value
 */
export function formatPressure(value: number, unit: string = 'bar'): string {
  return `${value.toFixed(1)} ${unit}`;
}

/**
 * Format temperature value
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
 * Format date
 */
export function formatDate(date: Date | string, format: string = 'short'): string {
  const d = typeof date === 'string' ? new Date(date) : date;

  switch (format) {
    case 'short':
      return d.toLocaleDateString();
    case 'long':
      return d.toLocaleDateString(undefined, {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
      });
    case 'time':
      return d.toLocaleTimeString();
    case 'datetime':
      return `${d.toLocaleDateString()} ${d.toLocaleTimeString()}`;
    case 'iso':
      return d.toISOString();
    default:
      return d.toLocaleDateString();
  }
}

/**
 * Format duration in milliseconds
 */
export function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) return `${days}d ${hours % 24}h`;
  if (hours > 0) return `${hours}h ${minutes % 60}m`;
  if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
  return `${seconds}s`;
}

/**
 * Format bytes
 */
export function formatBytes(bytes: number, decimals: number = 2): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(decimals))} ${sizes[i]}`;
}

/**
 * Format percentage
 */
export function formatPercent(value: number, decimals: number = 1): string {
  return `${value.toFixed(decimals)}%`;
}

/**
 * Format anomaly severity for display
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

/**
 * Truncate string with ellipsis
 */
export function truncate(str: string, maxLength: number): string {
  if (str.length <= maxLength) return str;
  return `${str.substring(0, maxLength - 3)}...`;
}

/**
 * Format relative time (e.g., "2 hours ago")
 */
export function formatRelativeTime(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffSeconds = Math.floor(diffMs / 1000);
  const diffMinutes = Math.floor(diffSeconds / 60);
  const diffHours = Math.floor(diffMinutes / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffDays > 0) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
  if (diffHours > 0) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
  if (diffMinutes > 0) return `${diffMinutes} minute${diffMinutes > 1 ? 's' : ''} ago`;
  return 'just now';
}

export function useFormatting() {
  return {
    formatPressure,
    formatTemperature,
    formatFlow,
    formatVibration,
    formatSensorValue,
    formatCurrency,
    formatDate,
    formatDuration,
    formatBytes,
    formatPercent,
    formatSeverity,
    formatSensorStatus,
    truncate,
    formatRelativeTime,
  };
}
