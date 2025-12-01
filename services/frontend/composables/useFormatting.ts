/**
 * Formatting Composable
 * Re-exports formatters from utils for composable pattern
 */

// Re-export all formatters from utils
export {
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
  formatNumber,
  formatNumberCompact,
  formatCurrencyCompact,
  formatPercentChange,
  capitalize,
  toTitleCase,
  snakeToTitle,
  camelToTitle,
} from '~/utils/formatters';

/**
 * Composable wrapper for formatters
 * Provides reactive access to formatting utilities
 */
export function useFormatting() {
  // Import all formatters
  const formatters = await import('~/utils/formatters');
  
  return {
    ...formatters,
  };
}
