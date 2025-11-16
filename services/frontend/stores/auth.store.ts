// Fixed auth store with proper nullable types and null safety
import type { User } from '~/types/api';

export const useAuthStore = defineStore('auth', () => {
  const user = ref<User | null>(null);
  const loading = ref(false);
  const error = ref<string | null>(null);

  const api = useApi();

  // Getters with null safety
  const isAuthenticated = computed(() => !!user.value);
  const userName = computed(() => {
    if (!user.value) return '';
    const u = user.value;
    if (u.first_name || u.last_name) {
      return `${u.first_name || ''} ${u.last_name || ''}`.trim();
    }
    return u.username || u.name || u.email || '';
  });

  // Actions with proper error handling
  const login = async (credentials: { email: string; password: string }) => {
    loading.value = true;
    error.value = null;

    try {
      const userData = await api.login(credentials);
      user.value = userData;
      return userData;
    } catch (err: any) {
      error.value = err?.data?.detail || 'Ошибка входа';
      throw err;
    } finally {
      loading.value = false;
    }
  };

  const register = async (userData: any) => {
    loading.value = true;
    error.value = null;

    try {
      const newUser = await api.register(userData);
      user.value = newUser;
      return newUser;
    } catch (err: any) {
      error.value = err?.data?.detail || 'Ошибка регистрации';
      throw err;
    } finally {
      loading.value = false;
    }
  };

  const logout = async () => {
    loading.value = true;
    try {
      await api.logout();
    } finally {
      user.value = null;
      loading.value = false;
      error.value = null;
    }
  };

  const fetchCurrentUser = async () => {
    // ✅ FIX: Add null safety for api.isAuthenticated
    if (!api?.isAuthenticated?.value) return null;

    loading.value = true;
    try {
      const userData = await api.getCurrentUser();
      user.value = userData;
      return userData;
    } catch (err: any) {
      if (err.status === 401) {
        user.value = null;
        await navigateTo('/auth/login');
      }
      throw err;
    } finally {
      loading.value = false;
    }
  };

  const updateProfile = async (profileData: Partial<User>) => {
    if (!user.value) throw new Error('User not logged in');

    loading.value = true;
    try {
      const updated = await api.updateUser(profileData);
      if (updated && user.value) {
        user.value = { ...user.value, ...updated };
      }
      return updated;
    } catch (err: any) {
      error.value = err?.data?.detail || 'Ошибка обновления профиля';
      throw err;
    } finally {
      loading.value = false;
    }
  };

  const initialize = async () => {
    if (process.server) return;

    // ✅ FIX: Add null safety for api.isAuthenticated (line 107)
    if (api?.isAuthenticated?.value) {
      try {
        await fetchCurrentUser();
      } catch {
        // Silent fail - user will need to login again
      }
    }
  };

  return {
    // State (not readonly for mutability)
    user,
    loading,
    error,

    // Getters
    isAuthenticated,
    userName,

    // Actions
    login,
    register,
    logout,
    fetchCurrentUser,
    updateProfile,
    initialize,
  };
});