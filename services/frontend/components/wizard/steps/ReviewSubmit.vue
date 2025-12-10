<script setup lang="ts">
import { ref, computed } from 'vue';
import type { GraphTopology, TopologySubmitResponse } from '~/types/gnn';
import { useTopology } from '~/composables/useTopology';

interface Props {
  topology: GraphTopology;
}

const props = defineProps<Props>();
const emit = defineEmits<{
  'submit-success': [response: TopologySubmitResponse];
  'submit-error': [error: string];
}>();

const { submitTopology, loading, error, submitResponse } = useTopology();
const isSubmitted = ref(false);

const componentsSummary = computed(() => {
  const types: Record<string, number> = {};
  props.topology.components.forEach((c) => {
    types[c.componentType] = (types[c.componentType] || 0) + 1;
  });
  return types;
});

const handleSubmit = async () => {
  try {
    const response = await submitTopology(props.topology);
    
    if (response.status === 'success') {
      isSubmitted.value = true;
      emit('submit-success', response);
    } else {
      emit('submit-error', response.message || 'Submission failed');
    }
  } catch (e: any) {
    emit('submit-error', e.message);
  }
};
</script>

<template>
  <div class="space-y-6">
    <div>
      <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
        Review & Submit
      </h3>
      <p class="text-sm text-gray-600 dark:text-gray-400">
        Review your topology configuration before submitting to GNN Service.
      </p>
    </div>

    <!-- Success State -->
    <Card
      v-if="isSubmitted && submitResponse"
      variant="elevated"
      class="bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800"
    >
      <div class="flex gap-4">
        <div class="shrink-0">
          <div class="h-12 w-12 rounded-full bg-green-100 dark:bg-green-900/50 flex items-center justify-center">
            <Icon name="heroicons:check-circle" class="h-8 w-8 text-green-600 dark:text-green-400" />
          </div>
        </div>
        <div class="flex-1">
          <h4 class="text-lg font-semibold text-green-900 dark:text-green-100 mb-2">
            Topology Submitted Successfully!
          </h4>
          <dl class="space-y-1 text-sm text-green-800 dark:text-green-200">
            <div>
              <dt class="inline font-medium">Topology ID:</dt>
              <dd class="inline ml-2">{{ submitResponse.topologyId }}</dd>
            </div>
            <div>
              <dt class="inline font-medium">Equipment ID:</dt>
              <dd class="inline ml-2">{{ submitResponse.equipmentId }}</dd>
            </div>
            <div>
              <dt class="inline font-medium">Components:</dt>
              <dd class="inline ml-2">{{ submitResponse.componentsCount }}</dd>
            </div>
            <div>
              <dt class="inline font-medium">Edges:</dt>
              <dd class="inline ml-2">{{ submitResponse.edgesCount }}</dd>
            </div>
          </dl>
          <p class="mt-3 text-sm text-green-700 dark:text-green-300">
            {{ submitResponse.message }}
          </p>
        </div>
      </div>
    </Card>

    <!-- Error State -->
    <Card
      v-else-if="error"
      variant="elevated"
      class="bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800"
    >
      <div class="flex gap-4">
        <Icon name="heroicons:x-circle" class="h-6 w-6 text-red-600 dark:text-red-400 shrink-0" />
        <div>
          <h4 class="text-lg font-semibold text-red-900 dark:text-red-100 mb-1">
            Submission Failed
          </h4>
          <p class="text-sm text-red-800 dark:text-red-200">
            {{ error }}
          </p>
        </div>
      </div>
    </Card>

    <!-- Review State -->
    <template v-else>
      <!-- Equipment Summary -->
      <Card>
        <template #header>
          <h4 class="text-base font-semibold text-gray-900 dark:text-gray-100">
            Equipment Information
          </h4>
        </template>

        <dl class="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div>
            <dt class="text-sm font-medium text-gray-500 dark:text-gray-400">Equipment ID</dt>
            <dd class="mt-1 text-sm text-gray-900 dark:text-gray-100">{{ topology.equipmentId }}</dd>
          </div>
          <div>
            <dt class="text-sm font-medium text-gray-500 dark:text-gray-400">Equipment Name</dt>
            <dd class="mt-1 text-sm text-gray-900 dark:text-gray-100">{{ topology.equipmentName }}</dd>
          </div>
          <div v-if="topology.equipmentType">
            <dt class="text-sm font-medium text-gray-500 dark:text-gray-400">Type</dt>
            <dd class="mt-1 text-sm text-gray-900 dark:text-gray-100">{{ topology.equipmentType }}</dd>
          </div>
          <div v-if="topology.operatingHours">
            <dt class="text-sm font-medium text-gray-500 dark:text-gray-400">Operating Hours</dt>
            <dd class="mt-1 text-sm text-gray-900 dark:text-gray-100">{{ topology.operatingHours }}h</dd>
          </div>
        </dl>
      </Card>

      <!-- Components Summary -->
      <Card>
        <template #header>
          <div class="flex items-center justify-between">
            <h4 class="text-base font-semibold text-gray-900 dark:text-gray-100">
              Components
            </h4>
            <Badge>{{ topology.components.length }} total</Badge>
          </div>
        </template>

        <div class="space-y-3">
          <div v-for="(count, type) in componentsSummary" :key="type" class="flex items-center justify-between">
            <span class="text-sm text-gray-700 dark:text-gray-300">{{ type.replace(/_/g, ' ') }}</span>
            <Badge variant="secondary" size="sm">{{ count }}</Badge>
          </div>
        </div>
      </Card>

      <!-- Edges Summary -->
      <Card>
        <template #header>
          <div class="flex items-center justify-between">
            <h4 class="text-base font-semibold text-gray-900 dark:text-gray-100">
              Connections
            </h4>
            <Badge>{{ topology.edges.length }} total</Badge>
          </div>
        </template>

        <div class="space-y-2">
          <div
            v-for="(edge, index) in topology.edges"
            :key="index"
            class="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300"
          >
            <Icon name="heroicons:arrow-right" class="h-4 w-4 text-gray-400" />
            <span>{{ edge.sourceId }}</span>
            <Icon name="heroicons:arrow-right" class="h-4 w-4 text-primary-600" />
            <span>{{ edge.targetId }}</span>
            <Badge variant="secondary" size="sm">{{ edge.edgeType }}</Badge>
          </div>
        </div>
      </Card>

      <!-- Submit Button -->
      <div class="flex justify-end">
        <Button
          size="lg"
          :loading="loading"
          :disabled="loading"
          @click="handleSubmit"
        >
          <Icon v-if="!loading" name="heroicons:paper-airplane" class="h-5 w-5 mr-2" />
          {{ loading ? 'Submitting...' : 'Submit Topology' }}
        </Button>
      </div>
    </template>
  </div>
</template>
