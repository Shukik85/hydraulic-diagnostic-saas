// @ts-check
let withNuxt = () => []

// Only try to import in non-typecheck environments
try {
  if (
    typeof process !== 'undefined' &&
    process.env.NODE_ENV !== 'test' &&
    !process.argv.some((a) => a.includes('typecheck'))
  ) {
    const { existsSync } = await import('fs')
    if (existsSync('./.nuxt/eslint.config.mjs')) {
      const nuxtEslint = await import('./.nuxt/eslint.config.mjs')
      withNuxt = (nuxtEslint as any).default || (nuxtEslint as any)
    }
  }
} catch {}

export default withNuxt(
  // Your custom configs here
)
