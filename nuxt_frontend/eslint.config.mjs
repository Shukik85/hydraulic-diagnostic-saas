// Plain ESM JS (no TS assertions here)
let withNuxt = () => []

try {
  if (
    typeof process !== 'undefined' &&
    process.env.NODE_ENV !== 'test' &&
    !process.argv.some((a) => a.includes('typecheck'))
  ) {
    const { existsSync } = await import('fs')
    if (existsSync('./.nuxt/eslint.config.mjs')) {
      const nuxtEslint = await import('./.nuxt/eslint.config.mjs')
      // Support both default and named export without TS assertions
      withNuxt = (nuxtEslint && nuxtEslint.default) ? nuxtEslint.default : nuxtEslint
      if (typeof withNuxt !== 'function') withNuxt = () => []
    }
  }
} catch {}

export default withNuxt(
  // Custom rules can be added here
)
