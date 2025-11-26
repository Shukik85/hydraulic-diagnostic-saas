/**
 * Authentication Composable
 * JWT-based authentication with token management
 */

import type { User } from '~/types';

interface LoginCredentials {
  email: string;
  password: string;
}

interface AuthTokens {
  accessToken: string;
  refreshToken: string;
  expiresAt: number;
}

interface AuthState {
  user: User | null;
  tokens: AuthTokens | null;
  isLoading: boolean;
  error: string | null;
}

const authState = ref<AuthState>({
  user: null,
  tokens: null,
  isLoading: false,
  error: null,
});

/**
 * Get tokens from localStorage
 */
function getStoredTokens(): AuthTokens | null {
  if (typeof window === 'undefined') return null;
  
  const stored = localStorage.getItem('auth_tokens');
  if (!stored) return null;
  
  try {
    return JSON.parse(stored);
  } catch {
    return null;
  }
}

/**
 * Save tokens to localStorage
 */
function saveTokens(tokens: AuthTokens): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem('auth_tokens', JSON.stringify(tokens));
}

/**
 * Clear tokens from localStorage
 */
function clearStoredTokens(): void {
  if (typeof window === 'undefined') return;
  localStorage.removeItem('auth_tokens');
}

/**
 * Check if token is expired
 */
function isTokenExpired(expiresAt: number): boolean {
  return Date.now() >= expiresAt;
}

export function useAuth() {
  const api = useApi();
  const router = useRouter();
  const toast = useToast();

  /**
   * Initialize auth state from storage
   */
  function init(): void {
    const tokens = getStoredTokens();
    if (tokens && !isTokenExpired(tokens.expiresAt)) {
      authState.value.tokens = tokens;
      // Load user profile
      loadUser();
    } else {
      clearStoredTokens();
    }
  }

  /**
   * Load user profile
   */
  async function loadUser(): Promise<void> {
    try {
      const response = await api.get<User>('/api/v1/auth/me');
      authState.value.user = response.data;
    } catch (error) {
      console.error('Failed to load user:', error);
      logout();
    }
  }

  /**
   * Login with credentials
   */
  async function login(credentials: LoginCredentials): Promise<void> {
    authState.value.isLoading = true;
    authState.value.error = null;

    try {
      const response = await api.post<{
        user: User;
        accessToken: string;
        refreshToken: string;
        expiresIn: number;
      }>('/api/v1/auth/login', credentials);

      const { user, accessToken, refreshToken, expiresIn } = response.data;

      const tokens: AuthTokens = {
        accessToken,
        refreshToken,
        expiresAt: Date.now() + expiresIn * 1000,
      };

      authState.value.user = user;
      authState.value.tokens = tokens;
      saveTokens(tokens);

      toast.success('Logged in successfully', 'Welcome back!');
      router.push('/app/dashboard');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Login failed';
      authState.value.error = message;
      toast.error(message, 'Login Error');
      throw error;
    } finally {
      authState.value.isLoading = false;
    }
  }

  /**
   * Logout
   */
  async function logout(): Promise<void> {
    try {
      if (authState.value.tokens) {
        await api.post('/api/v1/auth/logout', {
          refreshToken: authState.value.tokens.refreshToken,
        });
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      authState.value.user = null;
      authState.value.tokens = null;
      clearStoredTokens();
      toast.info('Logged out', '');
      router.push('/login');
    }
  }

  /**
   * Refresh access token
   */
  async function refreshToken(): Promise<boolean> {
    const currentTokens = authState.value.tokens;
    if (!currentTokens) return false;

    try {
      const response = await api.post<{
        accessToken: string;
        expiresIn: number;
      }>('/api/v1/auth/refresh', {
        refreshToken: currentTokens.refreshToken,
      });

      const { accessToken, expiresIn } = response.data;

      const newTokens: AuthTokens = {
        accessToken,
        refreshToken: currentTokens.refreshToken,
        expiresAt: Date.now() + expiresIn * 1000,
      };

      authState.value.tokens = newTokens;
      saveTokens(newTokens);

      return true;
    } catch (error) {
      console.error('Token refresh failed:', error);
      logout();
      return false;
    }
  }

  /**
   * Get current access token (auto-refresh if expired)
   */
  async function getAccessToken(): Promise<string | null> {
    const tokens = authState.value.tokens;
    if (!tokens) return null;

    // If token expires in less than 5 minutes, refresh it
    if (tokens.expiresAt - Date.now() < 5 * 60 * 1000) {
      const refreshed = await refreshToken();
      if (!refreshed) return null;
    }

    return authState.value.tokens?.accessToken || null;
  }

  // Computed properties
  const isAuthenticated = computed(() => !!authState.value.user && !!authState.value.tokens);
  const currentUser = computed(() => authState.value.user);
  const userRole = computed(() => authState.value.user?.role || null);
  const isAdmin = computed(() => authState.value.user?.role === 'admin');
  const isLoading = computed(() => authState.value.isLoading);
  const error = computed(() => authState.value.error);

  // Initialize on composable creation
  if (typeof window !== 'undefined') {
    init();
  }

  return {
    // State
    isAuthenticated,
    currentUser,
    userRole,
    isAdmin,
    isLoading,
    error,

    // Methods
    login,
    logout,
    refreshToken,
    getAccessToken,
  };
}
