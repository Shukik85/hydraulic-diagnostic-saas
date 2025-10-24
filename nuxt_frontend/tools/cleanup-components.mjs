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
  'components/figma',
  'components/layout', 
  'components/pages',
  'components/ui',
]

async function ensureDir(p) {
  await fs.mkdir(p, { recursive: true }).catch(() => {})
}

async function fileExists(filePath) {
  try {
    await fs.access(filePath)
    return true
  } catch {
    return false
  }
}

async function findAllTsxFiles() {
  const results = []
  
  async function scanDirectory(dir) {
    if (!await fileExists(dir)) {
      console.log(`Directory does not exist: ${dir}`)
      return
    }
    
    try {
      const entries = await fs.readdir(dir, { withFileTypes: true })
      
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name)
        
        if (entry.isDirectory()) {
          // Рекурсивно сканируем поддиректории
          await scanDirectory(fullPath)
        } else if (entry.isFile() && entry.name.endsWith('.tsx')) {
          results.push(fullPath)
        }
      }
    } catch (error) {
      console.error(`Error scanning directory ${dir}:`, error.message)
    }
  }
  
  // Сканируем все целевые директории
  for (const target of targets) {
    const targetPath = path.join(root, target)
    console.log(`Scanning for TSX files in: ${targetPath}`)
    await scanDirectory(targetPath)
  }
  
  // Также сканируем корневую components директорию на всякий случай
  const componentsRoot = path.join(root, 'components')
  if (await fileExists(componentsRoot)) {
    console.log(`Scanning for TSX files in: ${componentsRoot}`)
    await scanDirectory(componentsRoot)
  }
  
  return results
}

async function moveFile(file) {
  if (!await fileExists(file)) {
    console.log('File does not exist, skipping:', file)
    return false
  }
  
  const rel = path.relative(root, file)
  const dest = path.join(trash, rel)
  
  await ensureDir(path.dirname(dest))
  
  try {
    await fs.rename(file, dest)
    console.log('✓ Moved file:', rel)
    return true
  } catch (error) {
    // Fallback: copy + delete
    try {
      const data = await fs.readFile(file)
      await fs.writeFile(dest, data)
      await fs.unlink(file)
      console.log('✓ Copied and deleted file:', rel)
      return true
    } catch (copyError) {
      console.error('✗ Failed to move file:', rel, copyError.message)
      return false
    }
  }
}

async function run() {
  console.log('Starting cleanup of React/TSX components...')
  console.log('Root directory:', root)
  console.log('Trash directory:', trash)
  
  // Проверяем структуру директорий
  console.log('\nChecking directory structure:')
  for (const target of targets) {
    const targetPath = path.join(root, target)
    const exists = await fileExists(targetPath)
    console.log(`- ${target}: ${exists ? 'EXISTS' : 'NOT FOUND'}`)
    
    if (exists) {
      try {
        const entries = await fs.readdir(targetPath)
        const tsxFiles = entries.filter(name => name.endsWith('.tsx'))
        const vueFiles = entries.filter(name => name.endsWith('.vue'))
        console.log(`  Files: ${entries.length} total, ${tsxFiles.length} TSX, ${vueFiles.length} Vue`)
        if (tsxFiles.length > 0) {
          console.log(`  TSX files: ${tsxFiles.join(', ')}`)
        }
      } catch (error) {
        console.log(`  Error reading directory: ${error.message}`)
      }
    }
  }
  
  await ensureDir(trash)
  
  // Находим все TSX файлы
  console.log('\nSearching for TSX files...')
  const filesToMove = await findAllTsxFiles()
  
  if (filesToMove.length === 0) {
    console.log('No TSX files found to move.')
    
    // Проверим, есть ли вообще какие-то файлы в components
    const componentsDir = path.join(root, 'components')
    if (await fileExists(componentsDir)) {
      console.log('\nContents of components directory:')
      try {
        const entries = await fs.readdir(componentsDir, { withFileTypes: true })
        for (const entry of entries) {
          const type = entry.isDirectory() ? 'DIR' : 'FILE'
          console.log(`  ${type} ${entry.name}`)
        }
      } catch (error) {
        console.log(`  Error reading components directory: ${error.message}`)
      }
    }
    
    return
  }
  
  console.log(`\nFound ${filesToMove.length} TSX files to move:`)
  filesToMove.forEach(file => {
    console.log(`- ${path.relative(root, file)}`)
  })
  
  let movedCount = 0
  for (const file of filesToMove) {
    const success = await moveFile(file)
    if (success) movedCount++
  }
  
  console.log(`\nCleanup completed. Successfully moved ${movedCount} out of ${filesToMove.length} files.`)
}

run().catch((err) => {
  console.error('[cleanup-components] Error:', err)
  process.exit(1)
})