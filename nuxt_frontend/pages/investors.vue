<template>
  <div class="space-y-8">
    <!-- Header -->
    <div class="u-flex-between">
      <div>
        <h1 class="u-h2">Business Analytics</h1>
        <p class="u-body text-gray-600 dark:text-gray-400 mt-1">
          Key performance indicators and metrics for stakeholders
        </p>
      </div>
      <div class="flex items-center gap-3">
        <button @click="handlePrintReport" class="u-btn u-btn-secondary u-btn-md">
          <Icon name="heroicons:document-text" class="w-4 h-4 mr-2" />
          Export Report
        </button>
        <button class="u-btn u-btn-primary u-btn-md">
          <Icon name="heroicons:presentation-chart-line" class="w-4 h-4 mr-2" />
          Presentation
        </button>
      </div>
    </div>

    <!-- Key Business Metrics -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
      <div class="u-metric-card">
        <div class="u-metric-header">
          <h3 class="u-metric-label">Monthly Revenue</h3>
          <div class="u-metric-icon bg-blue-100 dark:bg-blue-900/30">
            <Icon name="heroicons:currency-dollar" class="w-5 h-5 text-blue-600 dark:text-blue-400" />
          </div>
        </div>
        <div class="u-metric-value">$2.4M</div>
        <div class="u-metric-change u-metric-change-positive mt-2">
          <Icon name="heroicons:arrow-trending-up" class="w-4 h-4" />
          <span>+23.5% MoM</span>
        </div>
      </div>

      <div class="u-metric-card">
        <div class="u-metric-header">
          <h3 class="u-metric-label">Active Customers</h3>
          <div class="u-metric-icon bg-green-100 dark:bg-green-900/30">
            <Icon name="heroicons:users" class="w-5 h-5 text-green-600 dark:text-green-400" />
          </div>
        </div>
        <div class="u-metric-value">127</div>
        <div class="u-metric-change u-metric-change-positive mt-2">
          <Icon name="heroicons:arrow-trending-up" class="w-4 h-4" />
          <span>+23 new</span>
        </div>
      </div>

      <div class="u-metric-card">
        <div class="u-metric-header">
          <h3 class="u-metric-label">Customer Retention</h3>
          <div class="u-metric-icon bg-purple-100 dark:bg-purple-900/30">
            <Icon name="heroicons:heart" class="w-5 h-5 text-purple-600 dark:text-purple-400" />
          </div>
        </div>
        <div class="u-metric-value">94.3%</div>
        <div class="u-metric-change u-metric-change-positive mt-2">
          <Icon name="heroicons:arrow-trending-up" class="w-4 h-4" />
          <span>+5.1% improvement</span>
        </div>
      </div>

      <div class="u-metric-card">
        <div class="u-metric-header">
          <h3 class="u-metric-label">Systems Monitored</h3>
          <div class="u-metric-icon bg-orange-100 dark:bg-orange-900/30">
            <Icon name="heroicons:server-stack" class="w-5 h-5 text-orange-600 dark:text-orange-400" />
          </div>
        </div>
        <div class="u-metric-value">1,847</div>
        <div class="u-metric-change u-metric-change-positive mt-2">
          <Icon name="heroicons:arrow-trending-up" class="w-4 h-4" />
          <span>+31.8% growth</span>
        </div>
      </div>
    </div>

    <!-- Financial Overview -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <div class="u-card">
        <div class="p-6 border-b border-gray-200 dark:border-gray-700">
          <h3 class="u-h4">Financial Performance</h3>
          <p class="u-body text-gray-600 dark:text-gray-400 mt-1">
            Revenue metrics and projections
          </p>
        </div>
        <div class="p-6 space-y-6">
          <div class="u-flex-between">
            <span class="u-body text-gray-600 dark:text-gray-400">Monthly Revenue</span>
            <span class="font-semibold text-gray-900 dark:text-white">
              {{ formatCurrency(marketData.businessIntelligence.totalRevenue.current) }}
            </span>
          </div>
          <div class="u-flex-between">
            <span class="u-body text-gray-600 dark:text-gray-400">MRR</span>
            <span class="font-semibold text-gray-900 dark:text-white">
              {{ formatCurrency(marketData.businessIntelligence.monthlyRecurringRevenue.current) }}
            </span>
          </div>
          <div class="u-flex-between">
            <span class="u-body text-gray-600 dark:text-gray-400">ARR Projection</span>
            <span class="font-semibold text-gray-900 dark:text-white">
              {{ formatCurrency(marketData.businessIntelligence.annualRunRate.projected) }}
            </span>
          </div>
          <div class="u-flex-between">
            <span class="u-body text-gray-600 dark:text-gray-400">Forecast Confidence</span>
            <span class="font-semibold text-green-600 dark:text-green-400">
              {{ marketData.businessIntelligence.annualRunRate.confidence }}%
            </span>
          </div>
        </div>
      </div>

      <div class="u-card">
        <div class="p-6 border-b border-gray-200 dark:border-gray-700">
          <h3 class="u-h4">Customer Analytics</h3>
          <p class="u-body text-gray-600 dark:text-gray-400 mt-1">
            Customer acquisition and retention metrics
          </p>
        </div>
        <div class="p-6 space-y-6">
          <div class="u-flex-between">
            <span class="u-body text-gray-600 dark:text-gray-400">Total Customers</span>
            <span class="font-semibold text-gray-900 dark:text-white">
              {{ formatNumber(marketData.businessIntelligence.customerMetrics.totalCustomers) }}
            </span>
          </div>
          <div class="u-flex-between">
            <span class="u-body text-gray-600 dark:text-gray-400">New This Month</span>
            <span class="font-semibold text-green-600 dark:text-green-400">
              +{{ marketData.businessIntelligence.customerMetrics.newCustomers }}
            </span>
          </div>
          <div class="u-flex-between">
            <span class="u-body text-gray-600 dark:text-gray-400">Churn Rate</span>
            <span class="font-semibold text-gray-900 dark:text-white">
              {{ marketData.businessIntelligence.customerMetrics.churnRate }}%
            </span>
          </div>
          <div class="u-flex-between">
            <span class="u-body text-gray-600 dark:text-gray-400">Net Revenue Retention</span>
            <span class="font-semibold text-green-600 dark:text-green-400">
              {{ marketData.businessIntelligence.customerMetrics.netRevenueRetention }}%
            </span>
          </div>
          <div class="u-flex-between">
            <span class="u-body text-gray-600 dark:text-gray-400">Avg Contract Value</span>
            <span class="font-semibold text-gray-900 dark:text-white">
              {{ formatCurrency(marketData.businessIntelligence.customerMetrics.averageContractValue) }}
            </span>
          </div>
        </div>
      </div>
    </div>

    <!-- Operational Excellence -->
    <div class="u-card">
      <div class="p-6 border-b border-gray-200 dark:border-gray-700">
        <h3 class="u-h4">Operational Excellence</h3>
        <p class="u-body text-gray-600 dark:text-gray-400 mt-1">
          System performance and efficiency metrics
        </p>
      </div>
      <div class="p-6">
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div class="text-center p-6 bg-blue-50 dark:bg-blue-900/20 rounded-xl">
            <div class="text-3xl font-bold text-blue-600 dark:text-blue-400 mb-2">
              {{ formatNumber(marketData.businessIntelligence.operationalMetrics.systemsMonitored) }}
            </div>
            <div class="u-body font-medium text-gray-700 dark:text-gray-300">
              Systems Monitored
            </div>
            <div class="u-body-sm text-blue-600 dark:text-blue-400 mt-1">
              24/7 Real-time
            </div>
          </div>

          <div class="text-center p-6 bg-green-50 dark:bg-green-900/20 rounded-xl">
            <div class="text-3xl font-bold text-green-600 dark:text-green-400 mb-2">
              {{ marketData.businessIntelligence.operationalMetrics.uptimePercentage }}%
            </div>
            <div class="u-body font-medium text-gray-700 dark:text-gray-300">
              Uptime SLA
            </div>
            <div class="u-body-sm text-green-600 dark:text-green-400 mt-1">
              Industry Leading
            </div>
          </div>

          <div class="text-center p-6 bg-purple-50 dark:bg-purple-900/20 rounded-xl">
            <div class="text-3xl font-bold text-purple-600 dark:text-purple-400 mb-2">
              {{ marketData.businessIntelligence.operationalMetrics.predictiveAccuracy }}%
            </div>
            <div class="u-body font-medium text-gray-700 dark:text-gray-300">
              Prediction Accuracy
            </div>
            <div class="u-body-sm text-purple-600 dark:text-purple-400 mt-1">
              30-day forecast
            </div>
          </div>

          <div class="text-center p-6 bg-orange-50 dark:bg-orange-900/20 rounded-xl">
            <div class="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-2">
              {{ formatNumber(marketData.businessIntelligence.operationalMetrics.alertsProcessed) }}
            </div>
            <div class="u-body font-medium text-gray-700 dark:text-gray-300">
              Alerts Processed
            </div>
            <div class="u-body-sm text-orange-600 dark:text-orange-400 mt-1">
              This month
            </div>
          </div>

          <div class="text-center p-6 bg-teal-50 dark:bg-teal-900/20 rounded-xl">
            <div class="text-3xl font-bold text-teal-600 dark:text-teal-400 mb-2">
              {{ marketData.businessIntelligence.operationalMetrics.avgResponseTime }}s
            </div>
            <div class="u-body font-medium text-gray-700 dark:text-gray-300">
              Avg Response Time
            </div>
            <div class="u-body-sm text-teal-600 dark:text-teal-400 mt-1">
              Sub-second alerts
            </div>
          </div>

          <div class="text-center p-6 bg-indigo-50 dark:bg-indigo-900/20 rounded-xl">
            <div class="text-3xl font-bold text-indigo-600 dark:text-indigo-400 mb-2">
              {{ formatCurrency(marketData.businessIntelligence.operationalMetrics.maintenanceCostSavings) }}
            </div>
            <div class="u-body font-medium text-gray-700 dark:text-gray-300">
              Cost Savings
            </div>
            <div class="u-body-sm text-indigo-600 dark:text-indigo-400 mt-1">
              Per customer/year
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { definePageMeta } from '#imports'

definePageMeta({
  title: 'Business Analytics',
  layout: 'default' as const,
  middleware: ['auth']
})

// Business intelligence data
const marketData = ref({
  businessIntelligence: {
    totalRevenue: { current: 28800000, growth: 23.5, period: 'Month' },
    monthlyRecurringRevenue: { current: 2400000, growth: 18.2, period: 'Month' },
    annualRunRate: { current: 34560000, projected: 45000000, confidence: 85 },
    customerMetrics: {
      totalCustomers: 127,
      newCustomers: 23,
      churnRate: 2.1,
      netRevenueRetention: 118.3,
      averageContractValue: 186000,
      customerSatisfactionScore: 4.7
    },
    operationalMetrics: {
      systemsMonitored: 1847,
      uptimePercentage: 99.94,
      alertsProcessed: 12840,
      avgResponseTime: 1.2,
      predictiveAccuracy: 94.8,
      maintenanceCostSavings: 3200000
    }
  }
})

// Methods
const handlePrintReport = () => {
  if (typeof window !== 'undefined') {
    window.print()
  }
}

const formatCurrency = (amount: number) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0
  }).format(amount)
}

const formatNumber = (number: number) => {
  return new Intl.NumberFormat('en-US').format(number)
}
</script>
