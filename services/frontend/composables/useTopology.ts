import { ref } from 'vue';
import type { GraphTopology, TopologySubmitResponse } from '~/types/gnn';

export const useTopology = () => {
  const config = useRuntimeConfig();
  const apiBaseUrl = config.public.apiBaseUrl;

  const loading = ref(false);
  const error = ref<string | null>(null);
  const topology = ref<GraphTopology | null>(null);
  const submitResponse = ref<TopologySubmitResponse | null>(null);

  /**
   * Submit topology to GNN Service
   * POST /api/v1/topology
   */
  const submitTopology = async (data: GraphTopology): Promise<TopologySubmitResponse> => {
    loading.value = true;
    error.value = null;

    try {
      const response = await $fetch<TopologySubmitResponse>(`${apiBaseUrl}/api/v1/topology`, {
        method: 'POST',
        body: data,
        headers: {
          'Content-Type': 'application/json',
        },
      });

      submitResponse.value = response;
      
      if (response.status === 'success') {
        console.log('[Topology] Submitted successfully:', response.topologyId);
      }

      return response;
    } catch (e: any) {
      const errorMessage = e.data?.message || e.message || 'Failed to submit topology';
      error.value = errorMessage;
      console.error('[Topology] Submit error:', e);
      
      throw new Error(errorMessage);
    } finally {
      loading.value = false;
    }
  };

  /**
   * Fetch existing topology by ID
   * GET /api/v1/topology/{topologyId}
   */
  const fetchTopology = async (topologyId: string): Promise<GraphTopology> => {
    loading.value = true;
    error.value = null;

    try {
      const response = await $fetch<GraphTopology>(
        `${apiBaseUrl}/api/v1/topology/${topologyId}`,
        {
          method: 'GET',
        }
      );

      topology.value = response;
      console.log('[Topology] Fetched successfully:', topologyId);

      return response;
    } catch (e: any) {
      const errorMessage = e.data?.message || e.message || 'Failed to fetch topology';
      error.value = errorMessage;
      console.error('[Topology] Fetch error:', e);
      
      throw new Error(errorMessage);
    } finally {
      loading.value = false;
    }
  };

  /**
   * Reset state
   */
  const reset = () => {
    loading.value = false;
    error.value = null;
    topology.value = null;
    submitResponse.value = null;
  };

  return {
    // State
    loading,
    error,
    topology,
    submitResponse,

    // Methods
    submitTopology,
    fetchTopology,
    reset,
  };
};
