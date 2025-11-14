import { test, expect } from '@playwright/test'

test.describe('Diagnosis Flow - Error Handling', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/diagnosis/demo')
  })

  test('shows error state and retry button', async ({ page }) => {
    await page.click('text=Toggle Error')
    await expect(page.locator('.error-container')).toBeVisible()
    await expect(page.locator('.error-title')).toContainText('Интерпретация недоступна')
    await page.click('text=Повторить попытку')
    await expect(page.locator('.error-container')).not.toBeVisible()
  })

  test('shows network error with troubleshooting tips', async ({ page }) => {
    await page.click('text=Toggle Network')
    await expect(page.locator('.network-error')).toBeVisible()
    await expect(page.locator('.tips-list li')).toHaveCount(4)
    await page.click('text=Проверить снова')
  })

  test('handles loading state transitions', async ({ page }) => {
    await page.click('text=Toggle Loading')
    await expect(page.locator('.loading-spinner')).toBeVisible()
    await expect(page.locator('.loading-text')).toContainText('Анализ результатов')
    await page.click('text=Toggle Loading')
    await expect(page.locator('.loading-spinner')).not.toBeVisible()
  })
})
