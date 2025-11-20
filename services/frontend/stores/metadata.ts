import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useMetadataStore = defineStore('metadata', () => {
  const data = ref<any>(null)

  function processMatrix(matrix: number[][]) {
    for (let i = 0; i < matrix.length; i++) {
      for (let j = 0; j < matrix[i]!.length; j++) {
        if (matrix[i] && matrix[i]![j] !== undefined) {
          matrix[i]![j] = 1
        }
      }
    }
  }

  return {
    data,
    processMatrix
  }
})
