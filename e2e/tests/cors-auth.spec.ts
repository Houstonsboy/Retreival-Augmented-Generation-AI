/**
 * Real-browser check: when the logged-in app calls the backend,
 * does the preflight pass, is the Bearer token attached, and is the reply 200?
 */
import { test, expect, type Request } from '@playwright/test';
import { ENV } from '../env';

test('dashboard → history call passes CORS with a Bearer token', async ({ page }) => {
  const backendHost = new URL(ENV.backendUrl).host;
  const consoleCors: string[] = [];
  page.on('console', (m) => {
    if (/CORS|Access-Control/i.test(m.text())) consoleCors.push(m.text());
  });

  const failed: string[] = [];
  page.on('requestfailed', (r: Request) => {
    if (r.url().includes(backendHost)) failed.push(`${r.method()} ${r.url()} — ${r.failure()?.errorText}`);
  });

  // /research loads the conversation list via apiFetch → /api/conversations (advocate-only).
  const convReq = page.waitForRequest((r) => r.url().includes(`${backendHost}/api/conversations`) && r.method() === 'GET');
  const convRes = page.waitForResponse((r) => r.url().includes(`${backendHost}/api/conversations`) && r.request().method() === 'GET');

  await page.goto('/research');
  await expect(page).not.toHaveURL(/\/auth\/sign-in/); // session was restored

  const req = await convReq;
  expect(req.headers()['authorization'], 'apiFetch did not attach the Firebase ID token').toMatch(/^Bearer .+/);

  const res = await convRes;
  const body = await res.text();
  if (res.status() === 403 && /PROFILE_REQUIRED/.test(body)) {
    throw new Error('403 PROFILE_REQUIRED — signed in, but users/{uid} is missing. Run npm run create-user.');
  }
  expect(res.status(), `${res.status()} ${body.slice(0, 200)}`).toBe(200);
  expect(res.headers()['access-control-allow-origin']).toBe(ENV.frontendUrl);

  expect(failed, 'backend requests failed (often CORS)').toEqual([]);
  expect(consoleCors, 'browser logged CORS errors').toEqual([]);
});
