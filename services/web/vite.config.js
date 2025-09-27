import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The web app will be served by Nginx at / (gateway). In dev, we hit http://localhost:8080.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 }
})
