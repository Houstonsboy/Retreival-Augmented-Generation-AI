/**
 * Full advocate flow: dashboard → Open chat → ask → qretrieve → generate-legal-answer → rendered answer.
 * Uses data-testid hooks from patches/03 when present, otherwise the confirmed visible text.
 */
import { test, expect, type Page } from '@playwright/test';
import { ENV, CHAT_API, RETRIEVE_API, UI } from '../env';

const hasTestId = async (page: Page, id: string) => (await page.getByTestId(id).count()) > 0;

test('advocate asks a question and gets a Legal Analysis answer', async ({ page }) => {
  await page.goto('/dashboard');
  await expect(page).toHaveURL(/\/dashboard/);

  await page.getByRole('button', { name: 'Open chat' }).click();
  await expect(page).toHaveURL(/\/research/);

  // Wait for the sidebar call so we know auth + CORS are live before asking.
  await page.waitForResponse((r) => r.url().includes('/api/conversations') && r.request().method() === 'GET');

  const input = (await hasTestId(page, 'chat-input'))
    ? page.getByTestId('chat-input')
    : page.locator('textarea[placeholder^="Ask a legal question"]');
  await input.fill(ENV.question);

  const retrieve = page.waitForResponse((r) => RETRIEVE_API.test(r.url()) && r.request().method() === 'POST', { timeout: 90_000 });
  const answer = page.waitForResponse((r) => CHAT_API.test(r.url()) && r.request().method() === 'POST', { timeout: 110_000 });

  // handleKeyDown: Enter (no Shift) calls handleSubmit(). Use the button if the testid exists.
  if (await hasTestId(page, 'chat-submit')) {
    const btn = page.getByTestId('chat-submit');
    await expect(btn).toBeEnabled();
    await btn.click();
  } else {
    await input.press('Enter');
  }

  // Loading card appears immediately.
  const loading = (await hasTestId(page, 'chat-loading')) ? page.getByTestId('chat-loading') : page.getByText(UI.loading);
  await expect(loading).toBeVisible({ timeout: 10_000 });

  // Step 1: retrieval
  const r1 = await retrieve;
  await assertOk(r1, '/api/qretrieve');
  const r1body = await r1.json();
  expect(r1body.success, `qretrieve returned success=${r1body.success}`).toBeTruthy();

  // Step 2: answer generation
  const r2 = await answer;
  await assertOk(r2, '/api/generate-legal-answer');
  const r2body = await r2.json();
  expect(typeof r2body.answer, 'response has no "answer" string').toBe('string');
  expect(r2body.answer.trim().length).toBeGreaterThan(20);

  // UI settles: loading gone, no error card, Legal Analysis card rendered.
  await expect(loading).toBeHidden({ timeout: 30_000 });

  const errorCard = (await hasTestId(page, 'chat-error')) ? page.getByTestId('chat-error') : page.getByText(UI.chatError);
  if (await errorCard.isVisible().catch(() => false)) {
    throw new Error(`Chat shows "Request failed": ${await errorCard.innerText()}`);
  }

  const assistant = (await hasTestId(page, 'assistant-message'))
    ? page.getByTestId('assistant-message').last()
    : page.getByText(UI.assistantHeader).last();
  await expect(assistant).toBeVisible();

  // ChatView uses history.replaceState → URL changes to /chat/<id> without a navigation event.
  await expect(page).toHaveURL(/\/chat\/[\w-]+/, { timeout: 15_000 });

  await page.screenshot({ path: `test-results/chat-${ENV.target}.png`, fullPage: true });
});

async function assertOk(res: import('@playwright/test').Response, label: string) {
  if (res.ok()) return;
  const body = (await res.text()).slice(0, 300);
  if (res.status() === 403 && /PROFILE_REQUIRED/.test(body)) {
    throw new Error(`${label}: 403 PROFILE_REQUIRED — no users/{uid} doc. Run npm run create-user.`);
  }
  if (res.status() === 403) throw new Error(`${label}: 403 — profile type is not advocate/admin. ${body}`);
  if (res.status() === 503) throw new Error(`${label}: 503 — Firebase Admin not initialised on backend.`);
  throw new Error(`${label}: HTTP ${res.status()} ${body}`);
}
