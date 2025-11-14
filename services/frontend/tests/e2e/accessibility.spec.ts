import { test, expect } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

test.describe('Diagnosis Flow - Accessibility', () => {
  test('should not have accessibility violations', async ({ page }) => {
    await page.goto('/diagnosis/demo')
    await page.click('text=Warning')
    const accessibilityScanResults = await new AxeBuilder({ page })
      .analyze()
    expect(accessibilityScanResults.violations).toEqual([])
  })

  test('keyboard navigation works', async ({ page }) => {
    await page.goto('/diagnosis/demo')
    await page.keyboard.press('Tab')
    await page.keyboard.press('Tab')
    await page.keyboard.press('Enter')
    const focused = page.locator(':focus')
    await expect(focused).toBeVisible()
  })

  test('screen reader labels present', async ({ page }) => {
    await page.goto('/diagnosis/demo')
    await page.click('text=Warning')
    await expect(page.locator('[aria-label]')).not.toHaveCount(0)
    await expect(page.locator('[aria-expanded]')).toHaveCount(2)
  })
})
