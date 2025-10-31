// @ts-check
let withNuxt = () => []

try {
  // Only import if .nuxt directory exists (after nuxt prepare/build)
  const nuxtEslint = await import('./.nuxt/eslint.config.mjs')
  withNuxt = nuxtEslint.default
} catch (error) {
  // Fallback for typecheck before nuxt prepare
  console.info('Using fallback ESLint config (pre-build)')
  withNuxt = () => []
}

export default withNuxt(
  // Your custom configs here
)