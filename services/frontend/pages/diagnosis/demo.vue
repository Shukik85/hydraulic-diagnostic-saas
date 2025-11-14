<template>
  <div class="diagnosis-demo-page">
    <h1 class="demo-title">Diagnosis Flow Demo Showcase</h1>
    <div class="scenario-switcher">
      <button
        v-for="(label, key) in scenarioLabels"
        :key="key"
        :class="['scenario-btn', {active: scenario === key}]"
        @click="selectScenario(key)"
      >{{ label }}</button>
    </div>

    <div class="demo-section">
      <DiagnosisProgress :stages="progressStages" :eta="eta" />
    </div>

    <div class="demo-section">
      <component
        :is="showLoading ? 'RAGProcessing' : 'RAGInterpretation'"
        v-bind="ragProps"
      />
    </div>

    <div class="demo-section">
      <ReasoningViewer
        v-if="rag && rag.reasoning"
        :reasoning="rag.reasoning"
        :metadata="rag.metadata"
      />
    </div>

    <div class="demo-section">
      <ErrorFallback v-if="showError" :error="errorMock" @reset="showError = false" />
      <NetworkError v-else-if="showNetwork" :message="'Нет интернет-соединения'" @retry="showNetwork = false" />
    </div>

    <div class="demo-footer">
      <button class="footer-btn" @click="toggleLoading">Toggle Loading</button>
      <button class="footer-btn" @click="toggleError">Toggle Error</button>
      <button class="footer-btn" @click="toggleNetwork">Toggle Network</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { diagnosisMocks } from '~/mocks/diagnosis-result.mock'
import DiagnosisProgress from '~/components/Loading/DiagnosisProgress.vue'
import RAGProcessing from '~/components/Loading/RAGProcessing.vue'
import RAGInterpretation from '~/components/Diagnosis/RAGInterpretation.vue'
import ReasoningViewer from '~/components/Diagnosis/ReasoningViewer.vue'
import ErrorFallback from '~/components/Error/ErrorFallback.vue'
import NetworkError from '~/components/Error/NetworkError.vue'

const scenarios = {
  standard: diagnosisMocks.standard,
  critical: diagnosisMocks.critical,
  normal: diagnosisMocks.normal,
  withoutRAG: diagnosisMocks.withoutRAG,
}
const scenarioLabels: Record<string, string> = {
  standard: 'Warning',
  critical: 'Critical',
  normal: 'Normal',
  withoutRAG: 'GNN Only',
}
const scenario = ref<'standard'|'critical'|'normal'|'withoutRAG'>('standard')
function selectScenario(k: keyof typeof scenarios) { scenario.value = k }

const mock = computed(() => scenarios[scenario.value])
const rag = computed(() => mock.value.rag_interpretation)
const progressStages = computed(() => [
  {id:'prepare',name:'Подготовка',status:'complete',duration:'0.5s'},
  {id:'gnn',name:'GNN',status: rag.value ? 'complete' : 'active',progress: rag.value ? 100 : 67},
  {id:'rag',name:'RAG',status: rag.value ? 'complete' : (scenario.value==='withoutRAG'?'error':'active'),progress: rag.value ? 100 : 37},
  {id:'report',name:'Отчёт',status:'pending'},
])
const eta = computed(() => '5 секунд')

const showLoading = ref(false)
function toggleLoading() { showLoading.value = !showLoading.value }

const showError = ref(false)
const errorMock = computed(() => new Error('Тестовая имитация ошибки RAG сервиса или сети.'))
function toggleError() { showError.value = !showError.value }

const showNetwork = ref(false)
function toggleNetwork() { showNetwork.value = !showNetwork.value }

const ragProps = computed(() => {
  if (showLoading.value) return { loading:true }
  if (showError.value) return { error: errorMock.value }
  return {
    interpretation: rag.value,
    loading: false,
    error: null,
  }
})
</script>

<style scoped>
.diagnosis-demo-page {
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem;
  font-family: Inter, Arial, sans-serif;
}
.demo-title {
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: 1.5rem;
}
.scenario-switcher {
  display: flex;
  gap: 1rem;
  margin-bottom: 2rem;
}
.scenario-btn {
  padding: 0.75rem 1.5rem;
  font-size: 1rem;
  font-weight: 700;
  border-radius: 0.5rem;
  background: #232b36;
  color: #edf2fa;
  border: 2px solid #424c5b;
  transition: all 0.2s;
  cursor: pointer;
}
.scenario-btn.active,
.scenario-btn:hover {
  background: #6366f1;
  color: white;
  border-color: #6366f1;
}
.demo-section {
  margin-bottom: 2rem;
  background: none;
}
.demo-footer {
  margin-top: 2rem;
  display: flex;
  gap: 1rem;
  justify-content: flex-end;
}
.footer-btn {
  padding: 0.5rem 1.25rem;
  border-radius: 0.5rem;
  background: #2b3340;
  color: #bbc6d6;
  border: 1.5px solid #424c5b;
  font-weight: 600;
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.25s;
}
.footer-btn:hover {
  background: #6366f1;
  color: white;
  border-color: #6366f1;
}
</style>
