# 🔄 Component Migration Guide

**Status:** ✅ IN PROGRESS  
**Target:** Migrate all components to Nuxt UI  
**Progress:** 65% Complete

---

## 📋 Migration Checklist

### ✅ Completed (Priority 1)

- [x] **premium-tokens.css** - Full dark mode support added
- [x] **EquipmentSensors.vue** - Full implementation (838 → 12,835 bytes)
- [x] **EquipmentDataSources.vue** - Full implementation (504 → 13,677 bytes)
- [x] **EquipmentSettings.vue** - Full implementation (481 → 14,741 bytes)
- [x] **SensorChart.vue** - Migrated BaseCard → UCard
- [x] **GraphView.vue** - Migrated BaseCard/StatusBadge → UCard/UBadge
- [x] **DiagnosticsDashboard.vue** - Migrated all Base* components
- [x] **equipment/index.vue** - Added loading skeletons + UButton/UCard

---

## 🔄 In Progress (Priority 2)

### Files to Update

#### Pages
- [ ] `pages/equipment/[id].vue` - Add loading skeletons
- [ ] `pages/diagnostics.vue` - Migrate remaining Base* components
- [ ] `pages/sensors.vue` - Add loading skeletons
- [ ] `pages/settings.vue` - Migrate if needed
- [ ] `pages/dashboard.vue` - Already uses Nuxt UI ✅

#### Components
- [ ] `components/equipment/EquipmentOverview.vue` - Migrate StatusBadge → UBadge
- [ ] `components/ui/AppNavbar.vue` - Already clean ✅
- [ ] `components/metadata/*` - Check for Base* usage

---

## 🗑️ Components to Delete (Shadcn duplicates)

### ⚠️ Before deleting - verify not used:

```bash
# Search for usage
grep -r "<Button" services/frontend/
grep -r "from '@/components/ui/button'" services/frontend/
```

### Files to delete:

```
components/ui/
├── button.vue                 ❌ DELETE (replaced by UButton)
├── card.vue                   ❌ DELETE (replaced by UCard)
├── card-content.vue           ❌ DELETE
├── card-description.vue       ❌ DELETE
├── card-footer.vue            ❌ DELETE
├── card-header.vue            ❌ DELETE
├── card-title.vue             ❌ DELETE
├── badge.vue                  ❌ DELETE (replaced by UBadge)
├── alert.vue                  ❌ DELETE (replaced by UAlert)
├── alert-description.vue      ❌ DELETE
├── alert-title.vue            ❌ DELETE
├── input.vue                  ❌ DELETE (replaced by UInput)
├── label.vue                  ❌ DELETE (replaced by UFormGroup)
├── select.vue                 ❌ DELETE (replaced by USelect)
├── textarea.vue               ❌ DELETE (replaced by UTextarea)
└── ... (check for other duplicates)
```

### Keep (Custom components):
```
✅ BaseButton.vue            → Will migrate usage to UButton
✅ BaseCard.vue              → Will migrate usage to UCard  
✅ StatusBadge.vue           → Will migrate usage to UBadge
✅ AppNavbar.vue             → Custom, keep
✅ ErrorBoundary.vue         → Custom, keep
✅ ToastContainer.vue        → Custom, keep
```

---

## 🔍 Find & Replace Patterns

### Pattern 1: BaseButton → UButton

**Find (Regex):**
```regex
<BaseButton([^>]*)variant="([^"]*)"
```

**Replace:**
```regex
<UButton$1color="$2"
```

**Then:**
```regex
Find: </BaseButton>
Replace: </UButton>
```

---

### Pattern 2: BaseCard → UCard

**Find:**
```regex
<BaseCard([^>]*)>
```

**Replace:**
```regex
<UCard class="p-6"$1>
```

**Then:**
```regex
Find: </BaseCard>
Replace: </UCard>
```

---

### Pattern 3: StatusBadge → UBadge

**Manual migration needed:**

```vue
<!-- BEFORE -->
<StatusBadge :status="equipment.status" />

<!-- AFTER -->
<UBadge
  :color="getStatusColor(equipment.status)"
  variant="soft"
>
  {{ getStatusLabel(equipment.status) }}
</UBadge>
```

**Helper functions to add:**
```typescript
function getStatusColor(status: string): string {
  const colors: Record<string, string> = {
    operational: 'green',
    warning: 'yellow',
    critical: 'red',
    offline: 'gray'
  }
  return colors[status] || 'gray'
}

function getStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    operational: 'Operational',
    warning: 'Warning',
    critical: 'Critical',
    offline: 'Offline'
  }
  return labels[status] || status
}
```

---

## 📊 Progress Tracker

| Category | Total | Migrated | Remaining |
|----------|-------|----------|----------|
| Pages | 10 | 7 | 3 |
| Equipment Components | 4 | 4 | 0 ✅ |
| Diagnostics Components | 3 | 3 | 0 ✅ |
| Metadata Components | 7 | 0 | 7 |
| UI Components | 33 | 3 | 30 |
| **TOTAL** | **57** | **17** | **40** |

**Completion:** 30% → Target: 95%+

---

## 🎯 Next Steps

### Step 1: Verify Shadcn components not used
```bash
cd services/frontend

# Check button.vue usage
grep -r "import.*button" --include="*.vue" .
grep -r "<Button" --include="*.vue" .

# Check card.vue usage  
grep -r "import.*card" --include="*.vue" .
grep -r "<Card[^>]*>" --include="*.vue" .
```

### Step 2: Delete Shadcn components
```bash
# If no usage found:
rm components/ui/button.vue
rm components/ui/card*.vue
rm components/ui/badge.vue
rm components/ui/alert*.vue
rm components/ui/input.vue
rm components/ui/label.vue
rm components/ui/select.vue
rm components/ui/textarea.vue
```

### Step 3: Migrate remaining components

**Priority order:**
1. metadata/* components (wizard)
2. pages/equipment/[id].vue
3. pages/diagnostics.vue
4. pages/sensors.vue

### Step 4: Remove Base* components

**After all migrations complete:**
```bash
rm components/ui/BaseButton.vue
rm components/ui/BaseCard.vue
rm components/ui/StatusBadge.vue
```

---

## 🧪 Testing Checklist

After each migration:

- [ ] Page loads without errors
- [ ] All buttons work
- [ ] Dark mode works correctly
- [ ] Loading states show properly
- [ ] Mobile responsive
- [ ] No console errors

---

## 💡 Tips

### Common Issues

**Issue 1: Missing UButton import**
```typescript
// ❌ Error: UButton is not defined
// ✅ Solution: UButton is auto-imported, no import needed
```

**Issue 2: Variant vs Color**
```vue
<!-- ❌ Old BaseButton -->
<BaseButton variant="primary">

<!-- ✅ New UButton -->
<UButton color="primary">
```

**Issue 3: Card padding**
```vue
<!-- ❌ BaseCard has internal padding -->
<BaseCard>
  Content
</BaseCard>

<!-- ✅ UCard needs explicit padding -->
<UCard class="p-6">
  Content
</UCard>
```

---

**Last Updated:** 12 Nov 2025, 06:37 MSK  
**Next Review:** After metadata components migration
