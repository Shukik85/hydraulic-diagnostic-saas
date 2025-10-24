/*
  Cleanup React/TSX leftovers from Nuxt app structure
  Run: node tools/cleanup-components.mjs

  What it does:
  - Moves TSX components and non-Nuxt dirs to nuxt_frontend/trash
  - Keeps only Vue/Nuxt components (*.vue) in components/
  - Safe operation: move (rename) with fallback copy+delete
*/

import { promises as fs } from 'node:fs'
import path from 'node:path'

const root = process.cwd()
const trash = path.join(root, 'trash')

// Targets to move (expandable)
const targets = [
  // React/TSX leftovers
  'components/figma/*.tsx',
  'components/layout/*.tsx',
  'components/pages/*.tsx',
  'components/ui/*.tsx',

  // Optional folders (uncomment if needed later)
  // 'design-system/**',
  // 'references/**',
]

async function ensureDir(p) {
  await fs.mkdir(p, { recursive: true }).catch(() => {})
}

async function listDir(dir) {
  try {
    return await fs.readdir(dir, { withFileTypes: true })
  } catch {
    return []
  }
}

function maskToRegex(mask) {
  // supports one-level masks like *.tsx
  return new RegExp('^' + mask.replace(/[.+^${}()|[\\]\\\\]/g, '\\$&').replace('*', '.*') + '$')
}

async function globSimple(pattern) {
  const [dirPart, mask] = pattern.split('/')
  const dirPath = path.join(root, dirPart)
  const entries = await listDir(dirPath)
  const regex = maskToRegex(mask)
  return entries
    .filter((e) => e.isFile() && regex.test(e.name))
    .map((e) => path.join(dirPath, e.name))
}

async function moveFile(file) {
  const rel = path.relative(root, file)
  const dest = path.join(trash, rel)
  await ensureDir(path.dirname(dest))
  try {
    await fs.rename(file, dest)
  } catch {
    const data = await fs.readFile(file)
    await fs.writeFile(dest, data)
    await fs.unlink(file)
  }
  console.log('Moved file:', rel)
}

async function moveDir(dir) {
  const rel = path.relative(root, dir)
  const dest = path.join(trash, rel)
  await ensureDir(path.dirname(dest))
  try {
    await fs.rename(dir, dest)
  } catch {
    // Fallback recursive copy then remove
    async function copyRecursive(src, dst) {
      await ensureDir(dst)
      const entries = await fs.readdir(src, { withFileTypes: true })
      for (const e of entries) {
        const s = path.join(src, e.name)
        const d = path.join(dst, e.name)
        if (e.isDirectory()) await copyRecursive(s, d)
        else await fs.copyFile(s, d)
      }
    }
    await copyRecursive(dir, dest)
    await fs.rm(dir, { recursive: true, force: true })
  }
  console.log('Moved dir:', rel)
}

async function run() {
  await ensureDir(trash)
  for (const pattern of targets) {
    if (pattern.endsWith('/**')) {
      const folder = path.join(root, pattern.replace('/**', ''))
      try {
        const stat = await fs.stat(folder)
        if (stat.isDirectory()) await moveDir(folder)
      } catch {}
      continue
    }
    const files = await globSimple(pattern)
    for (const file of files) {
      await moveFile(file)
    }
  }
  console.log('Cleanup completed successfully.')
}

run().catch((err) => {
  console.error('[cleanup-components] Error:', err)
  process.exit(1)
})