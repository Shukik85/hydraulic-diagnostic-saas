import { createApp } from 'vue'
import App from './App.vue'
import router from './router'  // Добавь эту строку
import './style.css'

const app = createApp(App)
app.use(router)  // Добавь эту строку
app.mount('#app')
