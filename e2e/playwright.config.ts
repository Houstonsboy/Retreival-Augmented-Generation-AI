import { defineConfig, devices } from '@playwright/test';
import { ENV, AUTH_FILE } from './env';

export default defineConfig({
  testDir: './tests',
  timeout: 120_000,              // RAG answers + Cloud Run cold start can be slow
  expect: { timeout: 15_000 },
  fullyParallel: false,          // one backend instance, keep it gentle
  workers: 1,
  retries: ENV.target === 'prod' ? 1 : 0,
  reporter: [['list'], ['html', { open: 'never' }]],

  use: {
    baseURL: ENV.frontendUrl,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },

  projects: [
    // 1. Pure HTTP checks against the backend — no browser, no login.
    { name: 'api', testMatch: /api\.spec\.ts/ },

    // 2. Log in once, save the Firebase session (incl. IndexedDB).
    { name: 'setup', testMatch: /auth\.setup\.ts/ },

    // 3. Browser tests reuse the saved session.
    {
      name: 'chromium',
      testMatch: /(cors-auth|chat)\.spec\.ts/,
      dependencies: ['setup'],
      use: { ...devices['Desktop Chrome'], storageState: AUTH_FILE },
    },
  ],
});
