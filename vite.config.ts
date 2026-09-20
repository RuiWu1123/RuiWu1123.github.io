import path from 'path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(() => {
    return {
      base: '/', // GitHub Pages 用户主页仓库使用根路径
      server: {
        port: 3000,
        host: '0.0.0.0',
      },
      plugins: [react()],
      define: {
        // Blog posts are plain files fetched at runtime, so the browser will
        // happily serve a stale copy after a deploy. Stamping each build lets
        // the fetch URL change whenever the site is rebuilt.
        __BUILD_ID__: JSON.stringify(Date.now().toString(36))
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      }
    };
});
