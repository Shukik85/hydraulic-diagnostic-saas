# 🛠️ Migration Guide - Frontend Production Fixes

> Пошаговая инструкция по применению исправлений

**Version:** 1.0.0 → 1.1.0  
**Date:** November 15, 2025  
**Estimated Time:** 15-20 minutes

---

## ✅ Pre-Migration Checklist

Перед началом убедись:

- [ ] Все изменения закоммичены
- [ ] Backend services работают (especially RAG service)
- [ ] Backup текущей ветки создан
- [ ] Node.js >= 20.x installed
- [ ] npm dependencies актуальны

---

## 🚀 Migration Steps

### Step 1: Backup Current State

```bash
# Create backup branch
git checkout master
git branch backup/pre-production-fixes-$(date +%Y%m%d)
git push origin backup/pre-production-fixes-$(date +%Y%m%d)

echo "✅ Backup created!"
```

---

### Step 2: Merge Fix Branch

```bash
# Fetch latest changes
git fetch origin

# Checkout and merge fix branch
git checkout master
git merge origin/feature/frontend-production-fixes

# Resolve conflicts if any (should be none)
```

**Ожидаемые изменения:**
```
 modified:   README.md
 modified:   app.vue
 modified:   composables/useGeneratedApi.ts
 new file:   .env.example
 new file:   composables/useRAG.ts
 new file:   composables/useMockData.ts
 new file:   types/rag.ts
 new file:   components/rag/InterpretationPanel.vue
 new file:   docs/ARCHITECTURE.md
 new file:   docs/RAG_INTEGRATION.md
 new file:   CHANGELOG.md
 new file:   MIGRATION_GUIDE.md
```

---

### Step 3: Delete Legacy Files

```bash
# Delete duplicate API composable
rm composables/useApi.ts

echo "✅ Legacy files removed!"
```

**Проверь:**
```bash
# Убедись, что файл удален
ls composables/ | grep useApi.ts
# Should return nothing
```

---

### Step 4: Update Environment Variables

```bash
# Copy example to .env
cp .env.example .env

# Edit with your values
nano .env  # or vim, code, etc.
```

**Обязательно настрой:**
```bash
# API Configuration
NUXT_PUBLIC_API_BASE=http://localhost:8000/api/v1
NUXT_PUBLIC_WS_BASE=ws://localhost:8000/ws

# Feature Flags
NUXT_PUBLIC_ENABLE_RAG=true              # ← Включи RAG!
NUXT_PUBLIC_ENABLE_WEBSOCKET=true
NUXT_PUBLIC_ENABLE_MOCK_DATA=false       # ← false для production

# Environment
NUXT_PUBLIC_ENVIRONMENT=development      # or production
```

**Проверь:**
```bash
cat .env | grep NUXT_PUBLIC_ENABLE_RAG
# Should show: NUXT_PUBLIC_ENABLE_RAG=true
```

---

### Step 5: Install Dependencies

```bash
# Clean install
rm -rf node_modules package-lock.json
npm install

echo "✅ Dependencies installed!"
```

**Проверь:**
```bash
npm list --depth=0
# Должны быть все зависимости
```

---

### Step 6: Generate API Client

```bash
# Generate TypeScript API client from OpenAPI spec
npm run generate:api

echo "✅ API client generated!"
```

**Проверь:**
```bash
ls generated/api/services/
# Should show: DiagnosisService.ts, EquipmentService.ts, GNNService.ts, RAGService.ts
```

---

### Step 7: Type Check

```bash
# Run TypeScript type checking
npm run typecheck

# Should pass without errors
```

**Ожидаемый результат:**
```
✅ Type checking complete - no errors!
```

**Если ошибки:**
- Проверь, что `useApi.ts` удален
- Проверь, что API client сгенерирован

---

### Step 8: Lint Code

```bash
# Run ESLint
npm run lint

# Auto-fix issues
npm run lint:fix
```

**Ожидаемый результат:**
```
✅ No linting errors!
```

---

### Step 9: Test Development Server

```bash
# Start dev server
npm run dev
```

**Проверь:**

1. **Открой:** http://localhost:3000
2. **Проверь консоль:**
   ```
   🚀 Hydraulic Diagnostic SaaS - development mode
   🛠️  Dev Mode:
     API Base: http://localhost:8000/api/v1
     Features: { ragInterpretation: true, ... }
   ```
3. **Нет ошибок** в console

---

### Step 10: Test RAG Features

```bash
# Test RAG composable
```

**Manual test:**

1. Открой `/diagnostics`
2. Запусти диагностику
3. Открой результат
4. Нажми "Генерировать интерпретацию"
5. Должно появиться: summary, analysis, recommendations

**Если RAG service не работает:**
- ✅ Должен показать fallback mode
- ✅ Не должно быть crash

---

### Step 11: Production Build Test

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

**Проверь:**
- ✅ Build успешен (no errors)
- ✅ Preview работает (http://localhost:3000)
- ✅ Bundle size < 300KB

---

### Step 12: Commit Changes

```bash
# Review changes
git status
git diff

# Stage deletion of useApi.ts
git add -u

# Commit
git commit -m "feat(frontend): Apply production fixes v1.1.0

- Merge feature/frontend-production-fixes
- Remove legacy useApi.ts
- Update .env with new variables
- Production-ready for accelerator demo

Breaking changes:
- useApi.ts removed (use useGeneratedApi instead)
- ENV variables renamed to NUXT_PUBLIC_*

See CHANGELOG.md for full details"

# Push
git push origin master

echo "✅ Migration complete!"
```

---

## 📝 Changes Summary

### ✨ Added (New Files)

```
.env.example
composables/useRAG.ts
composables/useMockData.ts
types/rag.ts
components/rag/InterpretationPanel.vue
docs/ARCHITECTURE.md
docs/RAG_INTEGRATION.md
CHANGELOG.md
MIGRATION_GUIDE.md (this file)
```

### 🔧 Modified (Updated Files)

```
README.md                         # Полностью переписан
app.vue                           # Error boundary added
composables/useGeneratedApi.ts    # Fixed imports
```

### ❌ Removed (Files to Delete)

```
composables/useApi.ts             # Удалить вручную!
```

---

## 🐛 Troubleshooting

### Issue 1: "Cannot find module useAuthStore"

**Symptom:**
```
ReferenceError: useAuthStore is not defined
```

**Fix:**
```bash
# Make sure useGeneratedApi.ts has the import
grep "import.*useAuthStore" composables/useGeneratedApi.ts

# Should show:
# import { useAuthStore } from '~/stores/auth.store'
```

---

### Issue 2: "RAG features not working"

**Symptom:**
RAG buttons не работают или не появляются.

**Check:**
```bash
# 1. Feature flag enabled?
grep ENABLE_RAG .env
# Should show: NUXT_PUBLIC_ENABLE_RAG=true

# 2. RAG service running?
curl http://localhost:8004/health
# Should return: {"status": "healthy"}

# 3. Check browser console
# Open DevTools → Console
# Should NOT show RAG-related errors
```

**Fix:**
```bash
# Enable RAG
echo "NUXT_PUBLIC_ENABLE_RAG=true" >> .env

# Start RAG service
cd ../../rag
docker-compose up -d

# Restart frontend
npm run dev
```

---

### Issue 3: "TypeScript errors after merge"

**Symptom:**
```
Type error: Cannot find name 'useApi'
```

**Fix:**
```bash
# Find all usages of old useApi
grep -r "useApi" --include="*.vue" --include="*.ts" .

# Replace with useGeneratedApi
# Should only find useGeneratedApi now

# Re-run type check
npm run typecheck
```

---

### Issue 4: "Build fails"

**Symptom:**
```
npm run build
# Error: ...
```

**Fix:**
```bash
# 1. Clean cache
rm -rf .nuxt node_modules
npm install

# 2. Regenerate API
npm run generate:api

# 3. Try again
npm run build
```

---

## 🔙 Rollback Plan

Если что-то пошло не так:

```bash
# 1. Checkout backup branch
git checkout backup/pre-production-fixes-YYYYMMDD

# 2. Force push to master (CAREFUL!)
git checkout master
git reset --hard backup/pre-production-fixes-YYYYMMDD
git push origin master --force

echo "✅ Rolled back to previous state"
```

**Или проще:**

```bash
# Revert merge commit
git revert -m 1 HEAD
git push origin master
```

---

## ✅ Post-Migration Checklist

После миграции проверь:

- [ ] Dev server запускается без ошибок
- [ ] TypeScript type check passes
- [ ] ESLint passes
- [ ] Production build succeeds
- [ ] RAG features работают (if enabled)
- [ ] No console errors в browser
- [ ] All pages load correctly
- [ ] README.md актуален
- [ ] Documentation complete
- [ ] `useApi.ts` deleted

---

## 📊 Testing Procedures

### Manual Testing

**Пройдись по критическим flow:**

1. **Landing → Login → Dashboard**
   - [ ] Landing page loads
   - [ ] Login works
   - [ ] Dashboard shows metrics

2. **Run Diagnostic**
   - [ ] Open /diagnostics
   - [ ] Click "Запустить"
   - [ ] Progress bar works
   - [ ] Result appears

3. **RAG Interpretation**
   - [ ] Open diagnostic result
   - [ ] Click "Генерировать интерпретацию"
   - [ ] Loading spinner shows
   - [ ] Interpretation appears with:
     - Summary
     - Analysis
     - Recommendations
     - Confidence score
     - Knowledge documents used

4. **Error Handling**
   - [ ] Trigger error (disconnect backend)
   - [ ] Error boundary catches it
   - [ ] User-friendly error page shows
   - [ ] "Try again" button works

---

### Automated Testing (Optional)

```bash
# Run unit tests
npm test

# Run E2E tests
npm run test:e2e
```

---

## 💼 For Accelerator Demo

### Pre-Demo Checklist

Перед презентацией акселератору:

- [ ] All services running (Django, GNN, RAG, TimescaleDB)
- [ ] Knowledge Base populated (20-50 documents)
- [ ] Mock data enabled for smooth demo (`ENABLE_MOCK_DATA=true`)
- [ ] Browser cache cleared
- [ ] Demo scenario prepared
- [ ] Backup slides ready

### Demo Flow

1. **Show Landing** (30 sec)
   - Modern UI, responsive
   - Clear value proposition

2. **Login → Dashboard** (30 sec)
   - Real-time metrics
   - System overview

3. **Run Diagnostic** (60 sec)
   - Select equipment
   - Show progress
   - Display results

4. **🎯 RAG Interpretation** (90 sec) **← WOW MOMENT!**
   - Click "Генерировать"
   - Show reasoning process
   - Highlight recommendations
   - Show knowledge base usage
   - **Emphasize AI differentiation!**

5. **Q&A** (remaining time)

---

## 📞 Support

**Проблемы с миграцией?**

- **Developer:** Plotnikov Aleksandr
- **Email:** shukik85@ya.ru
- **GitHub:** @Shukik85

**Документация:**
- [README.md](../README.md)
- [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [RAG_INTEGRATION.md](docs/RAG_INTEGRATION.md)
- [CHANGELOG.md](CHANGELOG.md)

---

**Удачной миграции!** 🚀

**Last Updated:** November 15, 2025