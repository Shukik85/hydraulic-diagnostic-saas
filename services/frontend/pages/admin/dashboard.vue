<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue';
import type { PlatformMetrics, RevenuePoint, TenantUsage, Alert } from '~/types';

definePageMeta({
  layout: 'admin',
  middleware: 'admin',
});

useSeoMeta({
  title: 'Admin Dashboard - Hydraulic Diagnostic SaaS',
  description: 'Platform metrics and analytics',
});

const metricsStore = useMetricsStore();
const { t } = useI18n();
const { formatCurrency, formatPercent } = useFormatting();

// Mock data for demonstration
const mockRevenueData = ref<RevenuePoint[]>([
  { date: 'Jan', value: 45000 },
  { date: 'Feb', value: 52000 },
  { date: 'Mar', value: 48000 },
  { date: 'Apr', value: 61000 },
  { date: 'May', value: 58000 },
  { date: 'Jun', value: 67000 },
  { date: 'Jul', value: 72000 },
]);

const mockTenants = ref<TenantUsage[]>([
  {
    tenantId: '1',
    tenantName: 'Acme Corp',
    sensors: 1234,
    apiCalls: 450000,
    storage: 52428800, // 50MB
    users: 45,
    plan: 'enterprise',
  },
  {
    tenantId: '2',
    tenantName: 'TechStart Inc',
    sensors: 567,
    apiCalls: 125000,
    storage: 20971520, // 20MB
    users: 12,
    plan: 'professional',
  },
  {
    tenantId: '3',
    tenantName: 'Industrial Solutions',
    sensors: 890,
    apiCalls: 230000,
    storage: 31457280, // 30MB
    users: 23,
    plan: 'professional',
  },
  {
    tenantId: '4',
    tenantName: 'Manufacturing Co',
    sensors: 234,
    apiCalls: 45000,
    storage: 10485760, // 10MB
    users: 8,
    plan: 'starter',
  },
  {
    tenantId: '5',
    tenantName: 'Hydraulics Plus',
    sensors: 456,
    apiCalls: 89000,
    storage: 15728640, // 15MB
    users: 15,
    plan: 'starter',
  },
]);

const mockAlerts = ref<Alert[]>([
  {
    id: '1',
    severity: 'critical',
    message: 'API latency exceeded 5s threshold',
    timestamp: new Date(Date.now() - 1800000), // 30 min ago
    source: 'API Gateway',
    resolved: false,
  },
  {
    id: '2',
    severity: 'high',
    message: 'Database connection pool near limit (85%)',
    timestamp: new Date(Date.now() - 3600000), // 1 hour ago
    source: 'PostgreSQL',
    resolved: false,
  },
  {
    id: '3',
    severity: 'medium',
    message: 'ML model prediction accuracy below 90%',
    timestamp: new Date(Date.now() - 7200000), // 2 hours ago
    source: 'GNN Service',
    resolved: false,
  },
]);

onMounted(async () => {
  // Subscribe to real-time metrics updates
  metricsStore.subscribeToMetrics();

  // Fetch initial data
  try {
    await metricsStore.refresh();
  } catch (error) {
    console.error('Failed to fetch metrics:', error);
  }
});

onUnmounted(() => {
  // Cleanup WebSocket connection
  metricsStore.unsubscribeFromMetrics();
});
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div>
      <h1 class="text-3xl font-bold text-gray-900 dark:text-white">
        {{ t('dashboard.title') }}
      </h1>
      <p class="mt-2 text-sm text-gray-600 dark:text-gray-400">
        {{ t('dashboard.subtitle') }}
      </p>
    </div>

    <!-- KPI Cards -->
    <div class="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
      <KpiCard
        :label="t('kpi.mrr')"
        :value="formatCurrency(metricsStore.metrics?.mrr || 72000)"
        :trend="metricsStore.metrics?.mrrGrowthPct || 12.5"
        icon="heroicons:currency-dollar"
        status="success"
        :subtext="t('kpi.vsLastMonth')"
        :loading="metricsStore.isLoading"
      />

      <KpiCard
        :label="t('kpi.tenants')"
        :value="metricsStore.metrics?.totalTenants || 1234"
        :trend="8.3"
        icon="heroicons:building-office"
        status="success"
        :subtext="`+${metricsStore.metrics?.newTenants30d || 89} ${t('kpi.last30Days')}`"
        :loading="metricsStore.isLoading"
      />

      <KpiCard
        :label="t('kpi.users')"
        :value="metricsStore.metrics?.totalUsers || 5678"
        icon="heroicons:users"
        status="neutral"
        :subtext="`+${metricsStore.metrics?.newUsers7d || 234} ${t('kpi.last7Days')}`"
        :loading="metricsStore.isLoading"
      />

      <KpiCard
        :label="t('kpi.uptime')"
        :value="formatPercent(metricsStore.metrics?.uptimePct ? metricsStore.metrics.uptimePct / 100 : 0.9995)"
        icon="heroicons:server"
        :status="metricsStore.isHealthy ? 'success' : 'warning'"
        :subtext="t('kpi.last30Days')"
        :loading="metricsStore.isLoading"
      />
    </div>

    <!-- Charts and Health -->
    <div class="grid gap-6 lg:grid-cols-2">
      <RevenueChart :data="metricsStore.revenueHistory.length > 0 ? metricsStore.revenueHistory : mockRevenueData" :loading="metricsStore.isLoading" />
      <SystemHealthPanel :health="metricsStore.metrics?.systemHealth || null" :loading="metricsStore.isLoading" />
    </div>

    <!-- Tenants and Alerts -->
    <div class="grid gap-6 lg:grid-cols-3">
      <div class="lg:col-span-2">
        <TenantsList :tenants="mockTenants" :loading="metricsStore.isLoading" />
      </div>
      <div>
        <AlertsList :alerts="mockAlerts" :loading="metricsStore.isLoading" />
      </div>
    </div>
  </div>
</template>
