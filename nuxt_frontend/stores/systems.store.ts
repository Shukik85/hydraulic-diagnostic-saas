interface HydraulicSystem {
    id: number
    name: string
    status: string
    pressure: number
    temperature: number
    lastUpdate: Date
}

export const useSystemsStore = defineStore('systems', () => {
    const systems = ref<HydraulicSystem[]>([])
    const isLoading = ref(false)
    const error = ref<string | null>(null)

    const fetchSystems = async () => {
        isLoading.value = true
        error.value = null

        try {
            const { $api } = useNuxtApp()
            systems.value = await $api<HydraulicSystem[]>('/systems/')
        } catch (err: any) {
            error.value = err.data?.detail || 'Failed to fetch systems'
        } finally {
            isLoading.value = false
        }
    }

    const fetchSystemById = async (id: number) => {
        try {
            const { $api } = useNuxtApp()
            return await $api<HydraulicSystem>(`/systems/${id}/`)
        } catch (err: any) {
            error.value = err.data?.detail || 'Failed to fetch system'
            return null
        }
    }

    const getSystemById = (id: number) => {
        return systems.value.find(s => s.id === id)
    }

    return {
        systems: readonly(systems),
        isLoading: readonly(isLoading),
        error: readonly(error),
        fetchSystems,
        fetchSystemById,
        getSystemById
    }
})