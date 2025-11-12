<template>
  <div class="metadata-wizard">
    <!-- ...Прочие шаги Wizard... -->
    <transition name="fade" mode="out-in">
      <component :is="currentLevelComponent" :key="currentLevel" />
    </transition>

    <!-- Level 6 шаг -->
    <transition name="fade" mode="out-in">
      <Level6SensorMapping v-if="currentLevel === 6" />
    </transition>

    <DataSourceModal v-if="showDataSourceModal" @go-csv="onGoCSV" @go-sim="onGoSim" @skip="onSkip" />

    <!-- ...Navigation footer... -->
  </div>
</template>

<script setup lang="ts">
import { useMetadataStore } from '~/stores/metadata'
import Level1BasicInfo from '~/components/metadata/Level1BasicInfo.vue'
import Level2GraphBuilder from '~/components/metadata/Level2GraphBuilder.vue'
import Level3ComponentForms from '~/components/metadata/Level3ComponentForms.vue'
import Level4DutyCycle from '~/components/metadata/Level4DutyCycle.vue'
import Level5Validation from '~/components/metadata/Level5Validation.vue'
import Level6SensorMapping from '~/components/metadata/Level6SensorMapping.vue'
import DataSourceModal from '~/components/setup/DataSourceModal.vue'

const store = useMetadataStore()
const levelNames = [
  'Базовая информация',
  'Архитектура системы',
  'Характеристики компонентов',
  'Профиль нагрузки',
  'Валидация',
  'Привязка датчиков'
]
const currentLevel = computed(() => store.wizardState.current_level)
const currentLevelComponent = computed(() => {
  const components = {
    1: Level1BasicInfo,
    2: Level2GraphBuilder,
    3: Level3ComponentForms,
    4: Level4DutyCycle,
    5: Level5Validation
  }
  return components[currentLevel.value as keyof typeof components]
})

const showDataSourceModal = ref(false)

// Навигация
function nextLevel() {
  if (currentLevel.value === 5) {
    store.completeLevel(5)
    store.goToLevel(6)
    return
  }
  if (currentLevel.value < 5) {
    store.completeLevel(currentLevel.value)
    store.goToLevel(currentLevel.value + 1)
  }
}
function previousLevel() {
  if (currentLevel.value > 1) {
    store.goToLevel(currentLevel.value - 1)
  }
}

// Управление DataSourceModal после Level 6
watch(
  () => currentLevel.value,
  (val) => {
    if (val === 6 && store.isSensorMappingComplete) {
      showDataSourceModal.value = true
    }
  }
)
function onGoCSV() {
  // навигация к CSVImportWizard или отдельной странице
  navigateTo('/import/csv')
}
function onGoSim() {
  // навигация к симулятору
  navigateTo('/simulator')
}
function onSkip() {
  // переход в system list/overview
  navigateTo('/equipment')
}
</script>
