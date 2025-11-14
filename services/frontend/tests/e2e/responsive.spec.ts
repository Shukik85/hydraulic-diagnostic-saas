import { test, expect, devices } from '@playwright/test'

test.describe('Diagnosis Flow - Responsive', () => {
  test('renders correctly on mobile', async ({ browser }) => {
    const context = await browser.newContext({
      ...devices['iPhone 13']
    })
    const page = await context.newPage()
    await page.goto('/diagnosis/demo')
    await page.click('text=Warning')
    await expect(page.locator('.rag-interpretation')).toBeVisible()
    const summaryCard = page.locator('.summary-card')
    const box = await summaryCard.boundingBox()
    expect(box?.width).toBeLessThan(500)
    await context.close()
  })

  test('renders correctly on tablet', async ({ browser }) => {
    const context = await browser.newContext({
      ...devices['iPad Pro']
    })
    const page = await context.newPage()
    await page.goto('/diagnosis/demo')
    await page.click('text=Critical')
    await expect(page.locator('.diagnosis-progress')).toBeVisible()
    await context.close()
  })
})
