/**
 * Capture the draft certificate UI screenshot.
 *
 * Strategy:
 *   1. Sign in with saved auth state.
 *   2. Navigate to /litigant/analysis.
 *   3. Intercept POST /sccvision/case/analyze → return a synthetic eligible
 *      response that points at the pre-seeded Firestore snapshot
 *      (analysis_id = SEEDED_ID).  The snapshot was written by
 *      scripts/seed_eligible_analysis.py.
 *   4. Upload a file + answer device questions → "analysis" resolves eligible.
 *   5. Screenshot step 2 showing "Prepare my certificate" button.
 *   6. Click through DeviceQuestionnaire → download PDF → screenshot step 3.
 *
 * Usage (from repo root):
 *   cd e2e && node screenshot_draft_cert.mjs
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
const BACKEND = "http://localhost:5000";

// The analysis_id seeded by scripts/seed_eligible_analysis.py
const SEEDED_ID = "2744c95f30424311a1272ecb49a10160";

// Synthetic eligible analyze response — matches the AnalystResponse interface
const MOCK_ANALYZE = {
  analysis_id: SEEDED_ID,
  prompt_version: "v2.2.0",
  verify_prompt_version: "v2.2.0",
  model: "llama-3.3-70b-versatile",
  calls_used: 2,
  trigger_fired: null,
  from_cache: false,
  round: 1,
  cert_eligibility: { eligible: true, reasons: [] },
  confidence: 0.92,
  verdict: "FAVOURABLE",
  integrity_rating: "ADEQUATE",
  derived_statuses: { A: "FAVOURABLE", B: "FAVOURABLE", C: "FAVOURABLE" },
  unstable_fields: [],
  model_disagreements: {},
  guided_follow_ups: [],
  analysis_input: {},
  result: {
    conversation_integrity: { rating: "ADEQUATE", issues: [] },
    unrelated_messages: [],
    parties: [
      { name: "Service Provider Ltd", role: "service_provider", notes: null },
      { name: "Client", role: "client", notes: null },
    ],
    timeline: [],
    elements: {
      A: {
        status: "FAVOURABLE",
        summary: "Service contract identified and confirmed between the parties.",
        evidence_ids: [],
        related_msg_indices: [],
        expected_effect: "Service was to be delivered as agreed.",
      },
      B: {
        status: "FAVOURABLE",
        summary: "Payment confirmed by declarant's device records.",
        evidence_ids: [],
        related_msg_indices: [],
        expected_effect: "Payment of KES 5,000 was made and acknowledged.",
      },
      C: {
        status: "FAVOURABLE",
        summary: "Follow-up communications confirm performance.",
        evidence_ids: [],
        related_msg_indices: [],
        expected_effect: "Service was provided within the agreed timeframe.",
      },
    },
    mpesa_payments: [],
    red_flags: [],
    missing_information: [],
    overall_verdict: "FAVOURABLE",
    confidence: 0.92,
    changes_since_previous: [],
    plain_summary:
      "The WhatsApp conversation shows a clear service agreement between the parties. " +
      "All elements of the contract have been identified, payment was confirmed, and " +
      "the evidence record is consistent and reliable. This evidence is suitable for " +
      "certification under Section 106B of the Evidence Act.",
    strengthen_checklist: [],
    follow_up_requests: [],
    questions_for_parties: [],
    disclaimer:
      "This automated analysis is not legal advice. Consult a qualified advocate before relying on it in court.",
  },
  verification: {
    quotes_grounded: true,
    ungrounded_quotes: [],
    mpesa_cross_check: {
      llm_payment_count: 0,
      parsed_payment_count: 0,
      codes_llm_only: [],
      codes_parsed_only: [],
      consistent: true,
      blocks_certificate: false,
      findings: [],
    },
    status_issues: [],
    premature_breach: null,
    verdict_before: "FAVOURABLE",
    verdict_after: "FAVOURABLE",
    verdict_downgraded: false,
    downgrade_reasons: [],
  },
};

fs.mkdirSync(FIGURES, { recursive: true });

async function ensureSignedIn(page) {
  await page.goto(`${FRONTEND}/litigant/certificates`, { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(3000);
  const url = page.url();
  if (!url.includes("/litigant")) {
    console.log("→ signing in fresh…");
    await page.goto(`${FRONTEND}/auth/sign-in`);
    await page.locator('input[type="email"]').fill("e2e-advocate@legalai.test");
    await page.locator('input[type="password"]').fill("E2eAdvocate!2026");
    await page.locator('button[type="submit"]').click();
    await Promise.race([
      page.waitForURL(/\/dashboard/, { timeout: 30_000 }),
      page.waitForURL(/\/litigant/, { timeout: 30_000 }),
    ]);
    console.log("  signed in →", page.url());
  } else {
    console.log("→ auth reused, on", url);
  }
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

  // ── Intercept the analyze endpoint ──────────────────────────────────────────
  await page.route(`${BACKEND}/sccvision/case/analyze`, async route => {
    console.log("  [intercept] /sccvision/case/analyze → returning mock eligible response");
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(MOCK_ANALYZE),
    });
  });

  // Also intercept the OCR extract and evidence/process endpoints to speed things up
  const FAKE_SHA = "aabbcc".repeat(10) + "aabb";
  await page.route(`${BACKEND}/sccvision/extract**`, async route => {
    console.log("  [intercept] /sccvision/extract → stub");
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        // fields that /evidence/process will read as ocr_result
        full_text: "Service confirmed. Payment received. Chat conversation between parties.",
        contact: {},
        messages: [{ sender: "Service Provider", text: "Payment received, service confirmed.", time: null, msg_index: 0 }],
        system_events: [],
        server_sha256: FAKE_SHA,
        word_count: 10,
      }),
    });
  });
  await page.route(`${BACKEND}/sccvision/evidence/process`, async route => {
    console.log("  [intercept] /sccvision/evidence/process → stub");
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        evidence: {
          evidence_id: "ev-synth-001",
          type: "chat_screenshot",
          file_name: "stitched_conversation.jpg",
          sha256: FAKE_SHA,
          server_sha256: FAKE_SHA,
          hash_match: true,
          added_at: new Date().toISOString(),
          ocr: {
            contact: {},
            messages: [{ sender: "Service Provider", text: "Payment received, service confirmed.", time: null, msg_index: 0 }],
            system_events: [],
          },
          parsed: null,
          user_edits: {},
        },
      }),
    });
  });

  // ── Analysis flow ───────────────────────────────────────────────────────────
  console.log("\n── Loading analysis page ──");
  await page.goto(`${FRONTEND}/litigant/analysis`, { waitUntil: "load" });
  await page.waitForTimeout(2000);

  // Step 0 — select role + consent
  const claimantBtn = page.locator('input[name="case-role"][value="claimant"]');
  if (await claimantBtn.count()) await claimantBtn.check();
  const consent = page.locator('input[type="checkbox"]').first();
  await consent.check();

  // Upload a file (just one — OCR will be intercepted)
  const evidenceFile = path.join(ROOT, "SCCImages", "stitched_conversation.jpg");
  const fileInput = page.locator('input[type="file"][multiple]');
  await fileInput.setInputFiles(evidenceFile);
  console.log("  uploaded stitched_conversation.jpg");
  await page.waitForTimeout(1000);

  // Click Continue
  await page.locator('button.scc-primary').last().click();
  console.log("  clicked Continue (OCR intercepted)");

  // Wait for step 1
  await page.waitForSelector('.scc-workshop[data-step="1"]', { timeout: 30_000 });
  await page.waitForTimeout(500);

  // Step 1 — answer device questions
  const questionGroups = page.locator('fieldset.scc-question');
  const qCount = await questionGroups.count();
  console.log(`  answering ${qCount} device question(s)…`);
  for (let i = 0; i < qCount; i++) {
    const firstRadio = questionGroups.nth(i).locator('input[type="radio"]').first();
    await firstRadio.check().catch(() => {});
  }
  await page.locator('button[type="submit"].scc-primary').click();
  console.log("  clicked Analyse (intercepted)");

  // Step 2 — wait for analysis result
  await page.waitForSelector('.scc-workshop[data-step="2"]', { timeout: 30_000 });
  await page.waitForTimeout(1500);

  const eligible = await page.locator('button:has-text("Prepare my certificate")').count() > 0;
  console.log("  cert_eligible:", eligible);

  if (!eligible) {
    console.error("ERROR: 'Prepare my certificate' button not found — mock intercept may have failed");
    // Screenshot for debug
    await page.screenshot({ path: path.join(FIGURES, "debug_step2.png") });
    await browser.close();
    process.exit(1);
  }

  // ── Screenshot step 2: eligible state ──────────────────────────────────────
  const out_step2 = path.join(FIGURES, "fig_draft_cert_eligible_step2.png");
  await page.screenshot({ path: out_step2, fullPage: false });
  console.log("\n  Fig — eligible step 2 saved →", out_step2);

  // ── Click Prepare my certificate → step 3 ──────────────────────────────────
  // Wait until the button is not disabled (analyzing=false)
  await page.waitForFunction(
    () => {
      const btn = [...document.querySelectorAll("button")]
        .find(b => b.textContent?.includes("Prepare my certificate"));
      return btn && !btn.disabled;
    },
    { timeout: 10_000 }
  );
  // Use JS click to bypass potential overlay/scroll issues
  await page.evaluate(() => {
    const btn = [...document.querySelectorAll("button")]
      .find(b => b.textContent?.includes("Prepare my certificate"));
    if (btn) btn.click();
    else throw new Error("Prepare my certificate button not found in evaluate");
  });
  console.log("  clicked Prepare my certificate (via JS)");

  // Wait briefly for step state change, then debug-screenshot
  await page.waitForTimeout(2000);
  const stepAttr = await page.locator('.scc-workshop').getAttribute('data-step').catch(() => "?");
  console.log("  data-step after click:", stepAttr);
  await page.screenshot({ path: path.join(FIGURES, "debug_after_click.png") });

  // Try to find the DeviceQuestionnaire by looking for the name input
  const nameInputVisible = await page.locator('input[autocomplete="name"]').count();
  console.log("  input[autocomplete=name] count:", nameInputVisible);

  if (!nameInputVisible) {
    // Maybe we need to wait longer for React re-render
    await page.waitForSelector(
      '[data-step="3"] form, input[autocomplete="name"], details.scc-device-checks',
      { timeout: 15_000 }
    );
  }
  await page.waitForTimeout(500);
  console.log("  reached step 3: DeviceQuestionnaire");

  // ── Fill DeviceQuestionnaire ────────────────────────────────────────────────
  // The form has these scc-field inputs in order:
  //   [0] declarant name  (autocomplete="name")
  //   [1] declarant position  (no autocomplete)
  //   [2] device make  (placeholder "e.g. Samsung")
  //   [3] device model  (placeholder "e.g. Galaxy A14")
  //   [4] phone last 4  (optional, placeholder "e.g. 6789")
  const formInputs = page.locator('form .scc-card input.scc-field');
  const inputCount = await formInputs.count();
  console.log(`  form scc-field inputs: ${inputCount}`);

  if (inputCount >= 1) await formInputs.nth(0).fill("Gift Karimi");          // name
  if (inputCount >= 2) await formInputs.nth(1).fill("Sole user of the device"); // position
  if (inputCount >= 3) await formInputs.nth(2).fill("Samsung");              // make
  if (inputCount >= 4) await formInputs.nth(3).fill("Galaxy A54");           // model
  // phone last 4 — skip (optional)

  // Production method dropdown
  const prodSelect = page.locator('form select').first();
  if (await prodSelect.count()) await prodSelect.selectOption("screenshot_on_device");

  // All "Yes" radios for device checks (inside details.scc-device-checks)
  const yesRadios = page.locator('details.scc-device-checks input[type="radio"][value="yes"]');
  const yesCount = await yesRadios.count();
  console.log(`  device check Yes radios: ${yesCount}`);
  for (let i = 0; i < yesCount; i++) await yesRadios.nth(i).check().catch(() => {});

  // Declaration checkbox
  const declarationCb = page.locator('input[type="checkbox"]').last();
  await declarationCb.check().catch(() => {});

  await page.waitForTimeout(500);

  // ── Screenshot the filled DeviceQuestionnaire ──────────────────────────────
  const out_form = path.join(FIGURES, "fig_draft_certificate_ui.png");
  await page.screenshot({ path: out_form, fullPage: false });
  console.log("  Fig — DeviceQuestionnaire (filled) saved →", out_form);

  // ── Submit → download PDF ──────────────────────────────────────────────────
  console.log("  submitting DeviceQuestionnaire (triggers backend gate + PDF)…");
  // Intercept the download certificate endpoint and redirect to a direct download
  // so Playwright's "download" event fires reliably (blob URLs can be missed)
  await page.route(`${BACKEND}/api/evidence/certificates/*/download`, async route => {
    console.log("  [intercept] certificate download → passthrough with Content-Disposition");
    const response = await route.fetch();
    const headers = { ...response.headers(), "Content-Disposition": "attachment; filename=\"SCC106B-draft.pdf\"" };
    await route.fulfill({ response, headers });
  });

  const [download] = await Promise.all([
    page.waitForEvent("download", { timeout: 60_000 }).catch(() => null),
    page.locator('button[type="submit"].scc-primary').click(),
  ]);

  // If no download event, check for validation error message on the page
  if (!download) {
    await page.waitForTimeout(3000);
    const errText = await page.locator('[role="alert"]').textContent().catch(() => "");
    console.error("  no download event. Error on page:", errText);
    await page.screenshot({ path: path.join(FIGURES, "debug_submit.png") });
    console.log("  debug screenshot saved. Attempting to continue anyway…");
  } else {
    const certPath = path.join(FIGURES, "draft_certificate.pdf");
    await download.saveAs(certPath);
    console.log("  draft certificate PDF saved →", certPath);
  }

  // Screenshot post-download state
  await page.waitForTimeout(1500);
  const out_done = path.join(FIGURES, "fig_draft_certificate_done.png");
  await page.screenshot({ path: out_done, fullPage: false });
  console.log("  Fig — post-download state saved →", out_done);

  await browser.close();
  console.log("\nDone. Screenshots in", FIGURES);
}

main().catch(err => { console.error(err); process.exit(1); });
