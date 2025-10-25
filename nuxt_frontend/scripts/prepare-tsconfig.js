#!/usr/bin/env node
const fs = require('fs')
const path = require('path')

// Ensure .nuxt directory exists
const nuxtDir = path.resolve('.nuxt')
if (!fs.existsSync(nuxtDir)) {
  fs.mkdirSync(nuxtDir, { recursive: true })
}

// Create tsconfig.app.json if it doesn't exist
const tsconfigAppPath = path.resolve('.nuxt/tsconfig.app.json')
if (!fs.existsSync(tsconfigAppPath)) {
  const tsconfigApp = {
    extends: '../tsconfig.json',
    include: [
      './**/*',
      '../**/*'
    ],
    exclude: [
      '../node_modules/**/*'
    ]
  }
  
  fs.writeFileSync(tsconfigAppPath, JSON.stringify(tsconfigApp, null, 2))
  console.log('✓ Created .nuxt/tsconfig.app.json')
}

// Create types directory if needed
const typesDir = path.resolve('.nuxt/types')
if (!fs.existsSync(typesDir)) {
  fs.mkdirSync(typesDir, { recursive: true })
}

console.log('✓ TypeScript config preparation completed')
