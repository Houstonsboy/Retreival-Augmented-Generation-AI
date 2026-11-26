/**
 * Dissertation screenshots script
 *
 * Captures:
 *   Fig 5.8  — Certificates page with outcome filter chips
 *   Fig 5.9  — Section 106B Deficiency Report (UI, not PDF)
 *   Fig 5.x  — Draft certificate PDF (all conditions met)
 *
 * Usage:
 *   cd /home/gift/Retreival-Augmented-Generation-AI
 *   node scripts/take_screenshots.mjs
 */

import { chromium } from "@playwright/test";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const FIGURES = path.join(ROOT, "figures");
const AUTH_FILE = path.join(ROOT, "e2e/.auth/advocate-local.json");
const FRONTEND = "http://localhost:3000";
const EVIDENCE_DIR = path.join(ROOT, "SCCImages");

const EVIDENCE_FILES = [
  "stitched_conversation.jpg",
  "WhatsApp Image 2026-11-08 at 20.11.54.jpeg",
  "WhatsApp Image 2026-11-08 at 20.12.06.jpeg",
];

fs.mkdirSync(FIGURES, { recursive: true });

async function ensureSignedIn(page) {
  await page.goto(`${FRONTEND}/litigant/certificates`, { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(3000);

  const url = page.url();
  if (url.includes("/auth/sign-in") || url === `${FRONTEND}/` || url.includes("/dashboard") === false && !url.includes("/litigant")) {
    console.log("→ signing in fresh…");
    await page.goto(`${FRONTEND}/auth/sign-in`);
    await page.locator('input[type="email"]').fill("e2e-advocate@legalai.test");
    await page.locator('input[type="password"]').fill("E2eAdvocate!2026");
    await page.locator('button[type="submit"]').click();
    // Advocate lands on /dashboard; litigant on /litigant/analysis
    await Promise.race([
      page.waitForURL(/\/dashboard/, { timeout: 30_000 }),
      page.waitForURL(/\/litigant/, { timeout: 30_000 }),
    ]);
    console.log("  signed in →", page.url());
  } else {
    console.log("→ auth state reused, on", url);
  }
}

async function screenshotFig58(page) {
  console.log("\n── Figure 5.8: Certificates page ──");
  await page.goto(`${FRONTEND}/litigant/certificates`, { waitUntil: "networkidle" });
  await page.waitForTimeout(2000);
  const out = path.join(FIGURES, "fig5_8_certificates_outcome_filters.png");
  await page.screenshot({ path: out, fullPage: false });
  console.log("  saved →", out);
}

async function runAnalysisAndScreenshot(page) {
  console.log("\n── Figure 5.9: Evidence Analysis + Deficiency Report ──");
  await page.goto(`${FRONTEND}/litigant/analysis`, { waitUntil: "networkidle" });
  await page.waitForTimeout(1500);

  // Step 0 — select role
  const claimantBtn = page.locator('input[name="case-role"][value="claimant"]');
  if (await claimantBtn.count()) await claimantBtn.click();

  // Tick consent
  const consent = page.locator('input[type="checkbox"]').first();
  await consent.check();

  // Upload evidence files
  const fileInput = page.locator('input[type="file"][multiple]');
  const existing = EVIDENCE_FILES
    .map(f => path.join(EVIDENCE_DIR, f))
    .filter(f => fs.existsSync(f));
  if (!existing.length) {
    console.warn("  No SCCImages found, uploading single follow-up image");
    const fallback = path.join(EVIDENCE_DIR, "followupwhatsapp.jpeg");
    if (fs.existsSync(fallback)) existing.push(fallback);
  }
  console.log("  uploading:", existing.map(f => path.basename(f)).join(", "));
  await fileInput.setInputFiles(existing);

  // Wait for OCR to process (the Continue button changes from spinner to arrow)
  await page.waitForSelector('.scc-primary:not([disabled])', { timeout: 60_000 });
  // Click Continue to go to device questions
  await page.locator('button.scc-primary').last().click();

  // Step 1 — device questions
  await page.waitForTimeout(1500);
  // Answer all device questions with the first option in each group
  const questionGroups = page.locator('.scc-card fieldset .scc-choices, fieldset .scc-choices');
  const count = await questionGroups.count();
  console.log(`  answering ${count} device questions…`);
  for (let i = 0; i < count; i++) {
    const firstRadio = questionGroups.nth(i).locator('input[type="radio"]').first();
    await firstRadio.check().catch(() => {});
  }
  // Click Analyse button
  await page.locator('button[type="submit"].scc-primary').click();

  // Step 2 — wait for analysis result (can take 30–90 s with LLM)
  console.log("  waiting for analysis (up to 120 s)…");
  await page.waitForSelector('.scc-workshop[data-step="2"]', { timeout: 120_000 });
  await page.waitForTimeout(1500);

  // Check eligibility
  const eligible = await page.locator('button:has-text("Prepare my certificate")').count() > 0;
  console.log("  cert_eligible:", eligible);

  if (!eligible) {
    // Scroll deficiency report into view and screenshot
    const defSel = '[aria-label="Section 106B deficiency report"]';
    const defBox = page.locator(defSel);
    if (await defBox.count() > 0) {
      await defBox.scrollIntoViewIfNeeded();
      await page.waitForTimeout(500);
    }
    const out59 = path.join(FIGURES, "fig5_9_s106b_deficiency_report.png");
    await page.screenshot({ path: out59, fullPage: false });
    console.log("  Fig 5.9 saved →", out59);
    return { eligible: false };
  }

  // --- eligible path: generate draft certificate ---
  console.log("  eligible → generating draft certificate");
  await page.locator('button:has-text("Prepare my certificate")').click();
  await page.waitForSelector('.scc-workshop[data-step="3"]', { timeout: 10_000 });
  await page.waitForTimeout(1000);

  // Fill DeviceQuestionnaire
  await page.locator('input[autocomplete="name"]').fill("Test Declarant");
  await page.locator('input[placeholder*="role"]').fill("Sole user of the device");
  await page.locator('input[placeholder="e.g. Samsung"]').fill("Samsung");
  await page.locator('input[placeholder="e.g. Galaxy A14"]').fill("Galaxy A54");
  // Production method
  await page.locator('select').selectOption("screenshot_on_device");
  // Device checks — answer all Yes
  const checkRadios = page.locator('.scc-device-checks input[type="radio"][value="yes"]');
  const checkCount = await checkRadios.count();
  for (let i = 0; i < checkCount; i++) await checkRadios.nth(i).check().catch(() => {});
  // Confirm declaration
  await page.locator('input[type="checkbox"]').last().check();
  // Download PDF
  const [download] = await Promise.all([
    page.waitForEvent("download", { timeout: 30_000 }),
    page.locator('button[type="submit"].scc-primary').click(),
  ]);
  const certPath = path.join(FIGURES, "draft_certificate.pdf");
  await download.saveAs(certPath);
  console.log("  draft certificate saved →", certPath);

  // Screenshot the "ready" state (step 3 confirmation)
  await page.waitForTimeout(1000);
  const outCert = path.join(FIGURES, "fig_draft_certificate_ui.png");
  await page.screenshot({ path: outCert, fullPage: false });
  console.log("  certificate UI saved →", outCert);
  return { eligible: true };
}

async function main() {
  console.log("Launching Chromium…");
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    storageState: AUTH_FILE,
    viewport: { width: 1280, height: 820 },
  });

  const page = await context.newPage();
  page.on("console", m => { if (m.type() === "error") console.error("  [browser]", m.text()); });

  await ensureSignedIn(page);

  // Figure 5.8 — certificates page (may be empty on first pass)
  await screenshotFig58(page);

  // Analysis flow — Figure 5.9 or draft certificate
  const { eligible } = await runAnalysisAndScreenshot(page);

  if (eligible) {
    // Now re-take Figure 5.8 — the new certificate should appear in the list
    console.log("\n── Re-taking Figure 5.8 (certificate now exists) ──");
    await page.goto(`${FRONTEND}/litigant/certificates`, { waitUntil: "networkidle" });
    await page.waitForTimeout(3000);
    const out = path.join(FIGURES, "fig5_8_certificates_outcome_filters.png");
    await page.screenshot({ path: out, fullPage: false });
    console.log("  re-saved →", out);
  }

  await browser.close();
  console.log("\nDone. Screenshots in", FIGURES);
}

main().catch(err => { console.error(err); process.exit(1); });
