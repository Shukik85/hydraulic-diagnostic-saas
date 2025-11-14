import { test, expect } from '@playwright/test'

test.describe('Diagnosis Flow - Happy Path', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/diagnosis/demo')
  })

  test('complete diagnosis with RAG interpretation', async ({ page }) => {
    await page.click('text=Warning')
    await expect(page.locator('.diagnosis-progress')).toBeVisible()
    await expect(page.locator('.rag-interpretation')).toBeVisible()
    await expect(page.locator('.summary-text')).toContainText('утечка давления')
    const recommendations = page.locator('.recommendation-item')
    await expect(recommendations).toHaveCount(4)
    const firstRec = recommendations.first()
    await expect(firstRec).toHaveClass(/priority-high/)
    await page.click('text=Процесс анализа')
    await expect(page.locator('.reasoning-content')).toBeVisible()
    const steps = page.locator('.step-item')
    await expect(steps).toHaveCount(3)
  })

  test('handles critical severity correctly', async ({ page }) => {
    await page.click('text=Critical')
    await expect(page.locator('.severity-critical')).toBeVisible()
    await expect(page.locator('.recommendation-item').first()).toContainText('НЕМЕДЛЕННО ОСТАНОВИТЬ')
  })

  test('handles GNN-only fallback', async ({ page }) => {
    await page.click('text=GNN Only')
    await expect(page.locator('.rag-interpretation .empty-container')).toBeVisible()
    await expect(page.locator('.stage-complete')).toBeVisible()
  })
})
