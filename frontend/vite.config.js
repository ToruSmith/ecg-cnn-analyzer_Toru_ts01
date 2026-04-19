import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/train': { target: 'http://localhost:8000', changeOrigin: true },
      '/predict': { target: 'http://localhost:8000', changeOrigin: true },
      '/report': { target: 'http://localhost:8000', changeOrigin: true },
      '/gradcam': { target: 'http://localhost:8000', changeOrigin: true },
      '/stop': { target: 'http://localhost:8000', changeOrigin: true },
    }
  }
})
