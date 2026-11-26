import { test as setup, expect } from '@playwright/test';
import fs from 'node:fs';
import { ENV, AUTH_FILE, UI } from '../env';

setup('sign in as advocate and save session', async ({ page }) => {
  if (!ENV.email || !ENV.password) {
    throw new Error('Set TEST_EMAIL and TEST_PASSWORD in .env (npm run create-user makes the account).');
  }
  fs.mkdirSync('.auth', { recursive: true });

  await page.goto('/auth/sign-in');
  await expect(page).toHaveURL(/\/auth\/sign-in/);

  await page.locator('input[type="email"]').fill(ENV.email);
  await page.locator('input[type="password"]').fill(ENV.password);
  await page.locator('button[type="submit"]').click();

  // sign-in/page.tsx always does router.replace("/dashboard") on success;
  // failures render friendlyError() text in a red banner.
  const e = UI.signInErrors;
  const outcome = await Promise.race([
    page.waitForURL(/\/dashboard/, { timeout: 30_000 }).then(() => 'ok' as const),
    page.getByText(e.badCreds).waitFor({ timeout: 30_000 }).then(() => 'creds' as const),
    page.getByText(e.noUser).waitFor({ timeout: 30_000 }).then(() => 'nouser' as const),
    page.getByText(e.rateLimited).waitFor({ timeout: 30_000 }).then(() => 'rate' as const),
    page.getByText(e.generic).waitFor({ timeout: 30_000 }).then(() => 'generic' as const),
  ]);

  const host = new URL(ENV.frontendUrl).host;
  switch (outcome) {
    case 'creds':   throw new Error('Firebase rejected TEST_PASSWORD. Rerun `npm run create-user` to reset it.');
    case 'nouser':  throw new Error('TEST_EMAIL has no Firebase Auth account. Run `npm run create-user`.');
    case 'rate':    throw new Error('Firebase rate-limited this account. Wait a few minutes.');
    case 'generic': throw new Error(
      `Generic sign-in error. Most likely cause: ${host} is not in Firebase → Authentication → Settings → Authorized domains ` +
      '(auth/unauthorized-domain is not mapped in friendlyError()). Check the browser console in the trace.');
  }

  await expect(page).toHaveURL(/\/dashboard/);
  // Litigant profiles get bounced to /litigant/analysis — the test user must be an advocate.
  await page.waitForTimeout(1_500);
  if (/\/litigant\//.test(page.url())) throw new Error('Test user profile has type "litigant"; it must be "advocate".');

  // Firebase JS SDK v9+ keeps the session in IndexedDB, not localStorage.
  // indexedDB: true (Playwright ≥1.51) captures it so later tests start logged in.
  await page.context().storageState({ path: AUTH_FILE, indexedDB: true });
});
