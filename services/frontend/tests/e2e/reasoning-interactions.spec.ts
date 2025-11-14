import { test, expect } from '@playwright/test'

test.describe('Reasoning Viewer - Interactions', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/diagnosis/demo')
    await page.click('text=Warning')
  })

  test('copy reasoning to clipboard', async ({ page }) => {
    await page.context().grantPermissions(['clipboard-read', 'clipboard-write'])
    await page.click('.action-btn >> nth=0')
    await expect(page.locator('.action-btn >> nth=0 >> [name="lucide:check"]')).toBeVisible({ timeout: 2000 })
    const clipboardText = await page.evaluate(() => navigator.clipboard.readText())
    expect(clipboardText).toContain('Анализ аномалии давления')
  })

  test('export reasoning as JSON', async ({ page }) => {
    const downloadPromise = page.waitForEvent('download')
    await page.click('.action-btn >> nth=1')
    const download = await downloadPromise
    expect(download.suggestedFilename()).toMatch(/reasoning-\d+\.json/)
  })

  test('expand/collapse raw reasoning', async ({ page }) => {
    await page.click('text=Полный текст reasoning')
    await expect(page.locator('.raw-text')).toBeVisible()
    await page.click('text=Полный текст reasoning')
    await expect(page.locator('.raw-text')).not.toBeVisible()
  })
})
