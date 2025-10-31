// @ts-check
let withNuxt = () => []

// Only try to import in non-typecheck environments
if (typeof process !== 'undefined' && 
    process.env.NODE_ENV !== 'test' && 
    !process.argv.includes('typecheck')) {
  try {
    const { existsSync } = await import('fs')
    if (existsSync('./.nuxt/eslint.config.mjs')) {
      const nuxtEslint = await import('./.nuxt/eslint.config.mjs')
      withNuxt = nuxtEslint.default || nuxtEslint
    }
  } catch (error) {
    // Silent fallback for typecheck/build environments
  }
}

export default withNuxt(
  // Your custom configs here
)