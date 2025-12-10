<template>
  <div class="ui-kit-container">
    <!-- Header -->
    <header class="kit-header">
      <h1>🎨 UI Kit Showcase</h1>
      <p>Vue 3 Composition API Components Library</p>
    </header>

    <div class="kit-content">
      <!-- Colors Section -->
      <section class="kit-section">
        <h2>Color Palette</h2>
        <div class="color-grid">
          <div class="color-card">
            <div class="color-swatch" style="background: var(--color-primary)"></div>
            <p class="color-name">Primary</p>
            <p class="color-code">#208680</p>
          </div>
          <div class="color-card">
            <div class="color-swatch" style="background: var(--color-success)"></div>
            <p class="color-name">Success</p>
            <p class="color-code">#22c55e</p>
          </div>
          <div class="color-card">
            <div class="color-swatch" style="background: var(--color-warning)"></div>
            <p class="color-name">Warning</p>
            <p class="color-code">#f59e0b</p>
          </div>
          <div class="color-card">
            <div class="color-swatch" style="background: var(--color-error)"></div>
            <p class="color-name">Error</p>
            <p class="color-code">#ef4444</p>
          </div>
        </div>
      </section>

      <!-- Buttons Section -->
      <section class="kit-section">
        <h2>Buttons</h2>
        <div class="demo-card">
          <h3>Primary Buttons</h3>
          <div class="flex gap-md">
            <button class="btn btn-primary">Primary</button>
            <button class="btn btn-primary btn-small">Small</button>
            <button class="btn btn-primary btn-large">Large</button>
            <button class="btn btn-primary" disabled>Disabled</button>
          </div>
        </div>

        <div class="demo-card">
          <h3>Secondary Buttons</h3>
          <div class="flex gap-md">
            <button class="btn btn-secondary">Secondary</button>
            <button class="btn btn-secondary btn-small">Small</button>
            <button class="btn btn-secondary btn-large">Large</button>
          </div>
        </div>

        <div class="demo-card">
          <h3>Status Buttons</h3>
          <div class="flex gap-md">
            <button class="btn btn-success">✓ Success</button>
            <button class="btn btn-danger">✕ Delete</button>
          </div>
        </div>
      </section>

      <!-- Forms Section -->
      <section class="kit-section">
        <h2>Forms & Inputs</h2>
        <div class="card">
          <form @submit.prevent="handleFormSubmit">
            <div class="form-group">
              <label for="name">Full Name</label>
              <input
                v-model="formData.name"
                type="text"
                id="name"
                placeholder="John Doe"
              />
            </div>

            <div class="form-group">
              <label for="email">Email</label>
              <input
                v-model="formData.email"
                type="email"
                id="email"
                placeholder="john@example.com"
              />
            </div>

            <div class="form-group">
              <label for="system">System Type</label>
              <select v-model="formData.systemType" id="system">
                <option value="">Select type...</option>
                <option value="industrial">Industrial</option>
                <option value="mobile">Mobile</option>
                <option value="marine">Marine</option>
                <option value="construction">Construction</option>
              </select>
            </div>

            <div class="form-group">
              <label for="notes">Notes</label>
              <textarea
                v-model="formData.notes"
                id="notes"
                placeholder="Add notes..."
              ></textarea>
            </div>

            <button type="submit" class="btn btn-primary">Submit</button>
          </form>
        </div>
      </section>

      <!-- Badges Section -->
      <section class="kit-section">
        <h2>Badges & Status</h2>
        <div class="demo-card">
          <div class="flex gap-md flex-wrap">
            <span class="badge badge-primary">Primary</span>
            <span class="badge badge-success">Active</span>
            <span class="badge badge-warning">Pending</span>
            <span class="badge badge-error">Failed</span>
          </div>
        </div>
      </section>

      <!-- Alerts Section -->
      <section class="kit-section">
        <h2>Alerts</h2>
        <div class="alert alert-success">
          <strong>Success!</strong> Your changes have been saved successfully.
        </div>
        <div class="alert alert-warning">
          <strong>Warning!</strong> Please review your input before proceeding.
        </div>
        <div class="alert alert-error">
          <strong>Error!</strong> An unexpected error occurred.
        </div>
        <div class="alert alert-info">
          <strong>Info:</strong> This is an informational message.
        </div>
      </section>

      <!-- Progress Section -->
      <section class="kit-section">
        <h2>Progress & Stats</h2>
        <div class="card">
          <h3>System Health</h3>
          <p class="text-muted">Overall Health Score</p>
          <div class="progress">
            <div class="progress-bar" :style="{ width: healthScore + '%' }"></div>
          </div>
          <p class="text-small text-muted">{{ healthScore }}%</p>

          <h3 style="margin-top: var(--spacing-lg)">Performance Metrics</h3>
          <div class="stats-grid">
            <div class="stat-card">
              <div class="stat-label">Pressure (bar)</div>
              <div class="stat-value">{{ systemMetrics.pressure }}</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Temperature (°C)</div>
              <div class="stat-value">{{ systemMetrics.temperature }}</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Uptime (%)</div>
              <div class="stat-value">{{ systemMetrics.uptime }}</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Active Systems</div>
              <div class="stat-value">{{ systemMetrics.activeSystems }}</div>
            </div>
          </div>
        </div>
      </section>

      <!-- Tables Section -->
      <section class="kit-section">
        <h2>Tables</h2>
        <div class="card">
          <table class="table">
            <thead>
              <tr>
                <th>System ID</th>
                <th>Type</th>
                <th>Status</th>
                <th>Health</th>
                <th>Last Update</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="system in systems" :key="system.id">
                <td>{{ system.id }}</td>
                <td>{{ system.type }}</td>
                <td>
                  <span
                    class="badge"
                    :class="'badge-' + statusClass(system.status)"
                  >
                    {{ system.status }}
                  </span>
                </td>
                <td>{{ system.health }}%</td>
                <td>{{ system.lastUpdate }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <!-- Tabs Section -->
      <section class="kit-section">
        <h2>Tabs</h2>
        <div class="card">
          <div class="tabs">
            <button
              v-for="(tab, index) in tabs"
              :key="index"
              class="tab"
              :class="{ active: activeTab === index }"
              @click="activeTab = index"
            >
              {{ tab }}
            </button>
          </div>
          <div class="tab-content">
            <p v-if="activeTab === 0">Overview tab content goes here.</p>
            <p v-else-if="activeTab === 1">Details tab content goes here.</p>
            <p v-else>Settings tab content goes here.</p>
          </div>
        </div>
      </section>

      <!-- Interactive State -->
      <section class="kit-section">
        <h2>Interactive State Example</h2>
        <div class="card">
          <h3>Counter: {{ count }}</h3>
          <div class="flex gap-md">
            <button class="btn btn-primary" @click="count--">Decrease</button>
            <button class="btn btn-primary" @click="count = 0">Reset</button>
            <button class="btn btn-primary" @click="count++">Increase</button>
          </div>
          <div class="progress mt-lg">
            <div class="progress-bar" :style="{ width: (count * 10) % 100 + '%' }"></div>
          </div>
        </div>
      </section>

      <!-- Cards Grid -->
      <section class="kit-section">
        <h2>Cards Grid</h2>
        <div class="card-grid">
          <div class="card">
            <h3>System Status</h3>
            <p class="text-muted mb-md">Overall system health is optimal.</p>
            <div class="flex flex-between">
              <span class="badge badge-success">Operational</span>
              <a href="#" class="link">View Details →</a>
            </div>
          </div>

          <div class="card">
            <h3>Recent Activity</h3>
            <p class="text-small text-muted mb-md">Last 7 days</p>
            <ul class="activity-list">
              <li>✓ System updated</li>
              <li>✓ Diagnostics completed</li>
              <li>⚠ Minor alert issued</li>
            </ul>
          </div>

          <div class="card">
            <h3>Quick Actions</h3>
            <div class="flex flex-col gap-sm">
              <button class="btn btn-secondary" style="width: 100%">Run Diagnostic</button>
              <button class="btn btn-secondary" style="width: 100%">Export Report</button>
              <button class="btn btn-secondary" style="width: 100%">View Analytics</button>
            </div>
          </div>
        </div>
      </section>
    </div>

    <!-- Footer -->
    <footer class="kit-footer">
      <p>UI Kit v2.0 — Vue 3 Composition API Components</p>
      <p>Hydraulic Diagnostic SaaS Platform</p>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed } from 'vue'

// Reactive state
const count = ref(0)
const activeTab = ref(0)
const healthScore = ref(85)

const formData = reactive({
  name: '',
  email: '',
  systemType: '',
  notes: '',
})

const systemMetrics = reactive({
  pressure: 150,
  temperature: 45,
  uptime: 99.9,
  activeSystems: 42,
})

const tabs = ref(['Overview', 'Details', 'Settings'])

const systems = ref([
  {
    id: '#SYS-001',
    type: 'Industrial',
    status: 'Active',
    health: 95,
    lastUpdate: '2 hours ago',
  },
  {
    id: '#SYS-002',
    type: 'Mobile',
    status: 'Active',
    health: 87,
    lastUpdate: '30 minutes ago',
  },
  {
    id: '#SYS-003',
    type: 'Construction',
    status: 'Maintenance',
    health: 72,
    lastUpdate: '5 days ago',
  },
  {
    id: '#SYS-004',
    type: 'Marine',
    status: 'Inactive',
    health: 45,
    lastUpdate: '30 days ago',
  },
])

// Methods
const handleFormSubmit = () => {
  alert(`Form submitted!\nName: ${formData.name}\nEmail: ${formData.email}`)
  // Reset form
  formData.name = ''
  formData.email = ''
  formData.systemType = ''
  formData.notes = ''
}

const statusClass = (status: string): string => {
  const map: Record<string, string> = {
    Active: 'success',
    Maintenance: 'warning',
    Inactive: 'error',
  }
  return map[status] || 'primary'
}
</script>

<style scoped>
/* Design System Variables */
:root {
  --color-primary: #208680;
  --color-primary-light: #32b8c6;
  --color-primary-dark: #1a6b68;
  --color-success: #22c55e;
  --color-warning: #f59e0b;
  --color-error: #ef4444;
  --color-info: #3b82f6;
  --color-bg: #f9fafb;
  --color-surface: #ffffff;
  --color-text: #111827;
  --color-text-secondary: #6b7280;
  --color-border: #e5e7eb;
  --spacing-xs: 0.25rem;
  --spacing-sm: 0.5rem;
  --spacing-md: 1rem;
  --spacing-lg: 1.5rem;
  --spacing-xl: 2rem;
  --spacing-2xl: 3rem;
  --radius-sm: 0.375rem;
  --radius-md: 0.5rem;
  --radius-lg: 0.75rem;
  --radius-full: 9999px;
  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}

.ui-kit-container {
  background-color: var(--color-bg);
  color: var(--color-text);
  min-height: 100vh;
}

/* Header */
.kit-header {
  background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-primary-dark) 100%);
  color: white;
  padding: var(--spacing-2xl) var(--spacing-xl);
  margin-bottom: var(--spacing-2xl);
  border-radius: var(--radius-lg);
}

.kit-header h1 {
  font-size: 2.5rem;
  margin-bottom: var(--spacing-md);
}

.kit-header p {
  font-size: 1.1rem;
  opacity: 0.95;
}

/* Content */
.kit-content {
  max-width: 1280px;
  margin: 0 auto;
  padding: var(--spacing-xl);
}

/* Sections */
.kit-section {
  margin-bottom: var(--spacing-2xl);
}

.kit-section h2 {
  font-size: 1.875rem;
  margin-bottom: var(--spacing-lg);
  border-left: 4px solid var(--color-primary);
  padding-left: var(--spacing-md);
}

.kit-section h3 {
  font-size: 1.25rem;
  margin-bottom: var(--spacing-md);
}

/* Grid */
.color-grid,
.stats-grid,
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: var(--spacing-lg);
}

/* Color Card */
.color-card {
  text-align: center;
}

.color-swatch {
  width: 100%;
  height: 120px;
  border-radius: var(--radius-lg);
  margin-bottom: var(--spacing-md);
  box-shadow: var(--shadow-md);
}

.color-name {
  font-weight: 600;
  margin-bottom: var(--spacing-xs);
}

.color-code {
  font-size: 0.875rem;
  color: var(--color-text-secondary);
  font-family: monospace;
}

/* Cards */
.card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: var(--spacing-lg);
  box-shadow: var(--shadow-sm);
  transition: all 0.3s ease;
}

.card:hover {
  box-shadow: var(--shadow-md);
  border-color: var(--color-primary);
}

.demo-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: var(--spacing-lg);
  margin-bottom: var(--spacing-lg);
}

/* Buttons */
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: var(--spacing-sm) var(--spacing-md);
  border: none;
  border-radius: var(--radius-md);
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  gap: var(--spacing-sm);
}

.btn-primary {
  background-color: var(--color-primary);
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background-color: var(--color-primary-dark);
  box-shadow: var(--shadow-md);
}

.btn-secondary {
  background-color: transparent;
  color: var(--color-primary);
  border: 2px solid var(--color-primary);
}

.btn-secondary:hover {
  background-color: rgba(32, 134, 128, 0.1);
}

.btn-success {
  background-color: var(--color-success);
  color: white;
}

.btn-danger {
  background-color: var(--color-error);
  color: white;
}

.btn-small {
  padding: var(--spacing-xs) var(--spacing-sm);
  font-size: 0.875rem;
}

.btn-large {
  padding: var(--spacing-md) var(--spacing-lg);
  font-size: 1.125rem;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Forms */
.form-group {
  margin-bottom: var(--spacing-lg);
}

label {
  display: block;
  margin-bottom: var(--spacing-sm);
  font-weight: 500;
  color: var(--color-text);
}

input,
textarea,
select {
  width: 100%;
  padding: var(--spacing-sm) var(--spacing-md);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  font-size: 1rem;
  font-family: inherit;
  transition: all 0.2s ease;
}

input:focus,
textarea:focus,
select:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(32, 134, 128, 0.1);
}

textarea {
  resize: vertical;
  min-height: 100px;
}

/* Badges */
.badge {
  display: inline-block;
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--radius-full);
  font-size: 0.875rem;
  font-weight: 600;
}

.badge-primary {
  background-color: rgba(32, 134, 128, 0.2);
  color: var(--color-primary-dark);
}

.badge-success {
  background-color: rgba(34, 197, 94, 0.2);
  color: #16a34a;
}

.badge-warning {
  background-color: rgba(245, 158, 11, 0.2);
  color: #b45309;
}

.badge-error {
  background-color: rgba(239, 68, 68, 0.2);
  color: #dc2626;
}

/* Alerts */
.alert {
  padding: var(--spacing-md) var(--spacing-lg);
  border-radius: var(--radius-md);
  border-left: 4px solid;
  margin-bottom: var(--spacing-lg);
}

.alert-success {
  background-color: rgba(34, 197, 94, 0.1);
  border-color: var(--color-success);
  color: #16a34a;
}

.alert-warning {
  background-color: rgba(245, 158, 11, 0.1);
  border-color: var(--color-warning);
  color: #b45309;
}

.alert-error {
  background-color: rgba(239, 68, 68, 0.1);
  border-color: var(--color-error);
  color: #dc2626;
}

.alert-info {
  background-color: rgba(59, 130, 246, 0.1);
  border-color: var(--color-info);
  color: #1e40af;
}

/* Tables */
.table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.95rem;
}

.table thead {
  background-color: var(--color-bg);
}

.table th {
  padding: var(--spacing-md) var(--spacing-lg);
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid var(--color-border);
}

.table td {
  padding: var(--spacing-md) var(--spacing-lg);
  border-bottom: 1px solid var(--color-border);
}

.table tbody tr:hover {
  background-color: var(--color-bg);
}

/* Tabs */
.tabs {
  display: flex;
  border-bottom: 2px solid var(--color-border);
  margin-bottom: var(--spacing-lg);
  gap: var(--spacing-lg);
}

.tab {
  padding: var(--spacing-md) 0;
  border: none;
  background: none;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 500;
  color: var(--color-text-secondary);
  transition: all 0.2s ease;
  position: relative;
}

.tab:hover {
  color: var(--color-text);
}

.tab.active {
  color: var(--color-primary);
}

.tab.active::after {
  content: '';
  position: absolute;
  bottom: -2px;
  left: 0;
  right: 0;
  height: 2px;
  background-color: var(--color-primary);
}

.tab-content {
  padding: var(--spacing-lg) 0;
}

/* Progress */
.progress {
  width: 100%;
  height: 8px;
  background-color: var(--color-border);
  border-radius: var(--radius-full);
  overflow: hidden;
}

.progress-bar {
  height: 100%;
  background: linear-gradient(90deg, var(--color-primary), var(--color-primary-light));
  border-radius: var(--radius-full);
  transition: width 0.3s ease;
}

/* Stats */
.stat-card {
  background: var(--color-surface);
  padding: var(--spacing-lg);
  border-radius: var(--radius-lg);
  border: 1px solid var(--color-border);
}

.stat-label {
  font-size: 0.875rem;
  color: var(--color-text-secondary);
  font-weight: 500;
  margin-bottom: var(--spacing-sm);
}

.stat-value {
  font-size: 2rem;
  font-weight: 700;
  color: var(--color-primary);
}

/* Utility Classes */
.flex {
  display: flex;
  gap: var(--spacing-md);
}

.flex-col {
  flex-direction: column;
}

.flex-wrap {
  flex-wrap: wrap;
}

.flex-between {
  justify-content: space-between;
}

.gap-sm {
  gap: var(--spacing-sm);
}

.gap-md {
  gap: var(--spacing-md);
}

.gap-lg {
  gap: var(--spacing-lg);
}

.text-muted {
  color: var(--color-text-secondary);
}

.text-small {
  font-size: 0.875rem;
}

.text-bold {
  font-weight: 600;
}

.mb-md {
  margin-bottom: var(--spacing-md);
}

.mb-lg {
  margin-bottom: var(--spacing-lg);
}

.mt-lg {
  margin-top: var(--spacing-lg);
}

.link {
  color: var(--color-primary);
  text-decoration: none;
  font-weight: 500;
  transition: color 0.2s ease;
}

.link:hover {
  color: var(--color-primary-dark);
}

.activity-list {
  list-style: none;
  padding: 0;
}

.activity-list li {
  padding: var(--spacing-sm) 0;
  color: var(--color-text);
}

/* Footer */
.kit-footer {
  max-width: 1280px;
  margin: 0 auto;
  padding: var(--spacing-xl);
  border-top: 1px solid var(--color-border);
  text-align: center;
  color: var(--color-text-secondary);
  font-size: 0.875rem;
}

/* Responsive */
@media (max-width: 768px) {
  .kit-header h1 {
    font-size: 1.875rem;
  }

  .color-grid,
  .stats-grid,
  .card-grid {
    grid-template-columns: 1fr;
  }

  .flex {
    flex-direction: column;
  }
}
</style>
