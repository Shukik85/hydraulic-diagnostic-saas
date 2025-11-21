# Frontend Composables

## useApi - Enterprise API Composable

### Features

✅ **JWT Refresh Rotation** - Автоматическое обновление access token через refresh token  
✅ **401 Interceptor** - Авто retry при истечении токена  
✅ **Request Queue** - Очередь запросов при refresh токена  
✅ **Type Safety** - Полная типизация TypeScript  
✅ **Cookie Storage** - Безопасное хранение токенов  
✅ **Error Handling** - Структурированная обработка ошибок  

---

## Usage

### Basic Auth

```typescript
// In Pinia store or component
const api = useApi()

// Login
const user = await api.login({
  email: 'user@example.com',
  password: 'secure_password'
})

// Get current user
const currentUser = await api.getCurrentUser()

// Update profile
const updated = await api.updateUser({
  first_name: 'John',
  last_name: 'Doe'
})

// Logout
await api.logout()
```

### Password Management

```typescript
const api = useApi()

// Change password
await api.changePassword({
  old_password: 'old_pass',
  new_password: 'new_secure_pass'
})

// Request password reset
await api.requestPasswordReset('user@example.com')

// Reset password with token
await api.resetPassword({
  token: 'reset_token_from_email',
  new_password: 'new_secure_pass'
})
```

### Authenticated Requests

```typescript
const api = useApi()

// Generic authenticated request
const data = await api.authenticatedFetch('/systems/list', {
  method: 'GET'
})

// With body
const result = await api.authenticatedFetch('/diagnostics/run', {
  method: 'POST',
  body: { system_id: 123 }
})
```

### Token Management

```typescript
const api = useApi()

// Check authentication status
if (api.isAuthenticated.value) {
  console.log('User is logged in')
}

// Get access token
const token = api.accessToken.value

// Manual token refresh
const newToken = await api.refreshAccessToken()

// Clear tokens (force logout)
api.clearTokens()
```

---

## How It Works

### JWT Refresh Flow

```
1. Request with expired access token
   ↓
2. Receive 401 response
   ↓
3. Check if refresh is already in progress
   ↓ (no)
4. Call /auth/refresh with refresh token
   ↓
5. Store new access & refresh tokens
   ↓
6. Retry original request with new token
   ↓
7. Return response to caller
```

### Request Queue

Если несколько запросов получают 401 одновременно:

1. Первый запрос начинает refresh
2. Остальные добавляются в очередь
3. После получения нового токена - все запросы повторяются

---

## Configuration

### Token Expiration

```typescript
// In composable (useApi.ts)
const accessTokenCookie = useCookie('access-token', {
  maxAge: 60 * 15,  // 15 minutes
  secure: process.env.NODE_ENV === 'production',
  sameSite: 'lax'
})

const refreshTokenCookie = useCookie('refresh-token', {
  maxAge: 60 * 60 * 24 * 7,  // 7 days
  secure: process.env.NODE_ENV === 'production',
  sameSite: 'lax',
  httpOnly: false  // Set to true if backend supports httpOnly
})
```

### Backend Endpoints

Требуемые API endpoints:

```
POST /auth/login
POST /auth/register
POST /auth/logout
POST /auth/refresh
GET  /auth/me
PATCH /auth/me
POST /auth/change-password
POST /auth/password-reset
POST /auth/password-reset/confirm
```

### Response Format

```typescript
// Login/Register response
{
  user: {
    id: number
    email: string
    first_name?: string
    last_name?: string
    // ...
  },
  tokens: {
    access: string
    refresh: string
  }
}

// Refresh response
{
  access: string
  refresh?: string  // Optional: new refresh token (rotation)
}

// User profile response
{
  id: number
  email: string
  // ... User fields
}
```

---

## Error Handling

```typescript
try {
  const user = await api.login(credentials)
} catch (error: any) {
  if (error.status === 401) {
    console.error('Неверные креденшелы')
  } else if (error.status === 429) {
    console.error('Too many requests')
  } else {
    console.error('Ошибка сервера:', error.message)
  }
}
```

---

## Security

### Best Practices

✅ **HTTPS only** - В production используй secure cookies  
✅ **Short-lived tokens** - Access token: 15 min, Refresh: 7 days  
✅ **Refresh rotation** - Новый refresh token при каждом refresh  
✅ **httpOnly cookies** - Защита от XSS  
✅ **SameSite** - Защита от CSRF  
✅ **Token invalidation** - Бэкенд должен валидировать refresh tokens  

---

## Testing

```typescript
// Mock for tests
const mockApi = {
  isAuthenticated: ref(true),
  accessToken: ref('mock_token'),
  login: vi.fn(),
  logout: vi.fn(),
  getCurrentUser: vi.fn().mockResolvedValue({ id: 1, email: 'test@example.com' }),
  // ...
}
```

---

## Troubleshooting

### "Токен не обновляется"

1. Проверь `/auth/refresh` endpoint
2. Проверь формат ответа (access, refresh)
3. Проверь cookies (есть ли refresh-token)

### "Бесконечный луп refresh"

1. Убедись, что `/auth/refresh` не возвращает 401
2. Проверь `isRefreshing` flag logic
3. Добавь timeout для refresh запроса

### "Cookies не сохраняются"

1. Проверь CORS настройки
2. Убедись, что `credentials: 'include'` включен
3. Проверь `sameSite` и `secure` настройки

---

## Related

- `stores/auth.store.ts` - Использует useApi
- `middleware/auth.ts` - Проверяет аутентификацию
- `types/api.ts` - Типы User, LoginCredentials, AuthTokens
