# 🔄 Component Migration Status

**Last Updated:** November 13, 2025, 22:42 MSK  
**Theme:** Metallic Industrial v1.0  
**Progress:** 4/60 components (6.7%)

---

## ✅ Migrated Components

### Core UI Components

| Component | Status | Commit | Notes |
|-----------|--------|--------|-------|
| `card.vue` | ✅ Done | `4cea49c` | Uses `card-metal` class |
| `button.vue` | ✅ Done | `76910d6` | Uses `btn-metal` + `btn-primary` |
| `input.vue` | ✅ Done | `660be59` | Uses `input-metal` class |
| `badge.vue` | ✅ Done | `5bd4f8c` | Added success/warning/error/info variants |

---

## 📋 Pending Migration

### High Priority (Core UI)

- [ ] `textarea.vue` - Similar to input
- [ ] `select.vue` - Dropdown with metallic styling
- [ ] `dialog.vue` - Modal with metal background
- [ ] `KpiCard.vue` - Dashboard metric cards
- [ ] `AppNavbar.vue` - Main navigation header

### Medium Priority (Form Elements)

- [ ] `checkbox.vue`
- [ ] `radio-group.vue`
- [ ] `switch.vue`
- [ ] `slider.vue`
- [ ] `label.vue`

### Medium Priority (Layout)

- [ ] `sidebar.vue`
- [ ] `separator.vue`
- [ ] `skeleton.vue`
- [ ] `progress.vue`

### Lower Priority (Advanced)

- [ ] `table.vue`
- [ ] `tabs.vue`
- [ ] `dropdown-menu.vue`
- [ ] `toast.vue`
- [ ] `alert.vue`
- [ ] Chart components (`chart-area`, `chart-bar`, `chart-line`)

### Custom Components

- [ ] `PremiumButton.vue`
- [ ] `SectionHeader.vue`
- [ ] `UModal.vue`
- [ ] `UCreateSystemModal.vue`
- [ ] `URunDiagnosticModal.vue`
- [ ] `UReportGenerateModal.vue`
- [ ] `UChatNewSessionModal.vue`
- [ ] `fab.vue`

### Dashboard Components

- [ ] `components/dashboard/*` (TBD)

### Digital Twin Components

- [ ] `components/digital-twin/*` (TBD)

### Metadata Components

- [ ] `components/metadata/*` (TBD)

---

## 🎨 Migration Patterns

### Pattern 1: Card Components

**Before:**
```vue
<div class="bg-gray-800 border border-gray-700 rounded-xl p-6">
```

**After:**
```vue
<div class="card-metal">
```

### Pattern 2: Button Components

**Before:**
```vue
<button class="bg-primary-600 hover:bg-primary-700 text-white">
```

**After:**
```vue
<button class="btn-metal btn-primary">
```

### Pattern 3: Input Components

**Before:**
```vue
<input class="bg-gray-800 border border-gray-700 focus:border-primary-500">
```

**After:**
```vue
<input class="input-metal">
```

### Pattern 4: Status Badges

**Before:**
```vue
<span class="bg-green-500 text-white px-2 py-1 rounded">
  Online
</span>
```

**After:**
```vue
<span class="badge-status badge-success">
  ● Online
</span>
```

---

## 🔧 Migration Guidelines

### Step 1: Identify Component

1. Check if component uses old color classes:
   - `bg-gray-800`, `bg-gray-900`
   - `bg-primary-500`, `bg-primary-600`
   - `border-gray-700`

### Step 2: Replace Classes

1. **Cards:** Replace with `card-metal`
2. **Buttons:** Replace with `btn-metal` (+ `btn-primary` for primary)
3. **Inputs:** Replace with `input-metal`
4. **Backgrounds:** Replace with semantic tokens:
   - `bg-gray-900` → `bg-background-primary`
   - `bg-gray-800` → `bg-background-secondary`
   - `bg-gray-700` → `bg-background-tertiary`

### Step 3: Update Colors

1. **Primary:** `#3b82f6` → `#4f46e5` (usually automatic via Tailwind)
2. **Text:** `text-gray-100` → `text-text-primary`
3. **Borders:** `border-gray-700` → `border-steel-medium`

### Step 4: Test

1. Visual check in browser
2. Dark mode compatibility
3. Hover/focus states
4. Responsive behavior

### Step 5: Commit

```bash
feat(ui): migrate ComponentName to metallic theme

- Replace old classes with metallic variants
- Update colors to industrial palette
- Maintain backward compatibility
```

---

## 📊 Migration Metrics

### By Priority

- **High Priority:** 0/5 (0%)
- **Medium Priority:** 0/15 (0%)
- **Lower Priority:** 0/20 (0%)
- **Custom Components:** 0/8 (0%)
- **Domain Components:** 0/12 (0%)

### By Type

- **Core UI:** 4/12 (33%)
- **Form Elements:** 0/8 (0%)
- **Layout:** 0/6 (0%)
- **Advanced:** 0/14 (0%)
- **Custom:** 0/8 (0%)
- **Domain-specific:** 0/12 (0%)

---

## 🎯 Next Steps

### Immediate (Today)

1. ✅ ~~Migrate core components (card, button, input, badge)~~
2. ⏳ Migrate form elements (textarea, select, checkbox)
3. ⏳ Migrate KpiCard for dashboard
4. ⏳ Migrate AppNavbar for consistent header

### Short-term (This Week)

1. Migrate all form components
2. Migrate layout components (sidebar, separator)
3. Migrate custom modal components
4. Update dashboard components

### Long-term (Next Week)

1. Migrate chart components
2. Migrate domain-specific components
3. Create component showcase page
4. Update Storybook (if exists)

---

## 💡 Tips

### Batch Migration

Migrate related components together:
- All form elements at once
- All modal components together
- All card variants together

### Testing Strategy

1. Create a test page with all migrated components
2. Check in both light and dark modes
3. Test responsive breakpoints
4. Verify accessibility

### Rollback Plan

If issues occur:
1. Old classes still work (Tailwind compatibility)
2. Can revert specific components
3. Git history preserved

---

## 🤝 Team Coordination

### Working On

- `@claude` - Core UI components (card, button, input, badge) ✅
- `@you` - TBD

### Code Review Checklist

- [ ] Uses metallic classes (card-metal, btn-metal, input-metal)
- [ ] Colors match industrial palette
- [ ] Focus states use indigo (#4f46e5)
- [ ] Hover effects smooth (0.2s transition)
- [ ] Backward compatible (old props still work)
- [ ] No breaking changes
- [ ] Visual test passed

---

**🎉 Keep going! Every migrated component brings us closer to a unified industrial aesthetic!**