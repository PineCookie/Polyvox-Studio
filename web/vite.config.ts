import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In dev, the API runs separately (`uv run voxcpm-studio`); in production the
// built UI is served by the API itself.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8808',
    },
  },
})
