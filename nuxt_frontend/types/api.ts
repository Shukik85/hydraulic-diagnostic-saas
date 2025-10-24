// types/api.ts
export interface User {
  id: number
  email: string
  name: string
}

export interface ChatSession {
  id: number
  title: string
  description: string
  lastMessage: string
  timestamp: string
  messages: ChatMessage[]
}

export interface ChatMessage {
  id: number
  role: string
  content: string
  timestamp: string
  sources?: { id: number; title: string }[]
}
