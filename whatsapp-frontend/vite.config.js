import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  base: '/gallery/',
  server: {
    // Allow requests from the unified gateway proxy
    cors: true,
    // Proxy /images requests to Strontium API
    proxy: {
      '/images': {
        target: 'http://localhost:5001',
        changeOrigin: true
      }
    }
  }
})
