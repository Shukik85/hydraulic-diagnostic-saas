#!/bin/bash
cd "$(dirname "$0")/.."

echo "🔧 Removing wrong imports from 'vue'..."

# Найти все файлы с неправильными импортами
find pages components -name "*.vue" | while read file; do
  # Удалить строки с импортами Nuxt API из 'vue'
  sed -i "/import.*\(definePageMeta\|useSeoMeta\|useRouter\|useAuthStore\|useI18n\).*from ['\"]vue['\"]/d" "$file"
  echo "✅ Fixed: $file"
done

echo "✨ Done!"
