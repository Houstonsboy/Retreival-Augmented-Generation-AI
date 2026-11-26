/**
 * Backend checks with no browser: health, CORS preflight, auth gate.
 * Run alone with:  npm run smoke
 */
import { test, expect } from '@playwright/test';
import { ENV } from '../env';

const origin = ENV.frontendUrl;
const api = (p: string) => `${ENV.backendUrl}${p}`;

test.describe('backend', () => {
  test('health endpoint is up', async ({ request }) => {
    // Generous timeout: first hit may be a Cloud Run cold start.
    const res = await request.get(api('/api/health'), { timeout: 90_000 });
    expect(res.status(), await res.text()).toBe(200);
  });

  // These are the exact endpoints the browser hits: sidebar load + both chat steps.
  for (const path of ['/api/conversations?limit=20', '/api/qretrieve', '/api/generate-legal-answer']) {
  test(`CORS preflight allows the frontend origin on ${path}`, async ({ request }) => {
    const method = path.startsWith('/api/conversations') ? 'GET' : 'POST';
    const res = await request.fetch(api(path), {
      method: 'OPTIONS',
      headers: {
        Origin: origin,
        'Access-Control-Request-Method': method,
        'Access-Control-Request-Headers': 'authorization,content-type',
      },
    });
    const h = res.headers();
    expect([200, 204]).toContain(res.status());
    expect(h['access-control-allow-origin'], 'CORS_ORIGIN on Cloud Run does not match the frontend URL').toBe(origin);
    expect((h['access-control-allow-headers'] ?? '').toLowerCase()).toMatch(/authorization|\*/);
    expect((h['access-control-allow-methods'] ?? '').toUpperCase()).toContain(method);
  });
  }

  test('CORS rejects an unknown origin', async ({ request }) => {
    const res = await request.fetch(api('/api/qretrieve'), {
      method: 'OPTIONS',
      headers: { Origin: 'https://evil.example.com', 'Access-Control-Request-Method': 'POST' },
    });
    expect(res.headers()['access-control-allow-origin']).not.toBe('https://evil.example.com');
  });

  test('protected route rejects requests without a token', async ({ request }) => {
    const res = await request.post(api('/api/qretrieve'), {
      headers: { Origin: origin, 'Content-Type': 'application/json' },
      data: { query: 'ping' },
    });
    // 503 here means Firebase Admin failed to initialise (FIREBASE_SERVICE_ACCOUNT_PATH / ADC).
    expect(res.status(), res.status() === 503 ? 'Firebase Admin not initialised on backend' : '').toBe(401);
  });

  test('protected route rejects a garbage token', async ({ request }) => {
    const res = await request.post(api('/api/qretrieve'), {
      headers: { Origin: origin, Authorization: 'Bearer not-a-real-token', 'Content-Type': 'application/json' },
      data: { query: 'ping' },
    });
    expect([401, 403]).toContain(res.status());
  });
});
