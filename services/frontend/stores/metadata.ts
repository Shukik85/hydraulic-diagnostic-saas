import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useMetadataStore = defineStore('metadata', () => {
  const data = ref<any>(null)

  function processMatrix(matrix: number[][]) {
    for (let i = 0; i < matrix.length; i++) {
      const row = matrix[i]
      if (!row) continue
      
      for (let j = 0; j < row.length; j++) {
        // Enterprise: безопасный доступ с проверкой
        if (row[j] !== undefined) {
          row[j] = 1
        }
      }
    }
  }

  return {
    data,
    processMatrix
  }
})
