import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/ask':        'http://localhost:8000',
      '/auth':       'http://localhost:8000',
      '/sensors':    'http://localhost:8000',
      '/events':     'http://localhost:8000',
      '/health':     'http://localhost:8000',
      '/setup-db':   'http://localhost:8000',
      '/eggs':       'http://localhost:8000',
      '/chores':     'http://localhost:8000',
      '/automation': 'http://localhost:8000',
      '/reviews':    'http://localhost:8000',
      '/heatmap':    'http://localhost:8000',
      '/risk':       'http://localhost:8000',
      '/weather':    'http://localhost:8000',
      '/static':     'http://localhost:8000',
    },
  },
})
