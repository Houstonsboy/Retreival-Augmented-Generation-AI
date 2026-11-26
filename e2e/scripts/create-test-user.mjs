#!/usr/bin/env node
/**
 * Creates (or repairs) the Playwright test account in legalai-scc:
 *   - Firebase Auth email/password user
 *   - users/{uid}            — exact 7-field shape enforced by firestore.rules (hasAll + hasOnly)
 *   - usernames/{username}   — { uid }
 *
 * Mirrors lib/firestore.ts:createUserDoc() as called from sign-up/page.tsx.
 * Admin SDK bypasses rules, but we keep the shape identical so the test user is indistinguishable
 * from a real sign-up.
 *
 *   npm i firebase-admin && npm run create-user
 */
import 'dotenv/config';
import fs from 'node:fs';
import { initializeApp, cert } from 'firebase-admin/app';
import { getAuth } from 'firebase-admin/auth';
import { getFirestore, FieldValue } from 'firebase-admin/firestore';

const { TEST_EMAIL, TEST_PASSWORD, TEST_USERNAME = 'e2e_advocate', FIREBASE_SERVICE_ACCOUNT_PATH } = process.env;

if (!TEST_EMAIL || !TEST_PASSWORD) throw new Error('Set TEST_EMAIL and TEST_PASSWORD in .env');
if (TEST_PASSWORD.length < 6) throw new Error('TEST_PASSWORD must be at least 6 characters (Firebase rule)');

const username = TEST_USERNAME.toLowerCase();
if (!/^[a-z0-9_]{3,30}$/.test(username)) {
  throw new Error(`TEST_USERNAME "${username}" must match ^[a-z0-9_]+$ and be 3–30 chars (firestore.rules)`);
}
if (!FIREBASE_SERVICE_ACCOUNT_PATH || !fs.existsSync(FIREBASE_SERVICE_ACCOUNT_PATH)) {
  throw new Error(`Service account not found at FIREBASE_SERVICE_ACCOUNT_PATH=${FIREBASE_SERVICE_ACCOUNT_PATH}`);
}

const sa = JSON.parse(fs.readFileSync(FIREBASE_SERVICE_ACCOUNT_PATH, 'utf8'));
if (sa.project_id !== 'legalai-scc') {
  throw new Error(`Service account is for "${sa.project_id}", expected "legalai-scc"`);
}
initializeApp({ credential: cert(sa) });
const auth = getAuth();
const db = getFirestore();

// 1. Auth user
let user;
try {
  user = await auth.getUserByEmail(TEST_EMAIL);
  await auth.updateUser(user.uid, { password: TEST_PASSWORD, emailVerified: true, disabled: false });
  console.log(`✓ Auth user exists: ${user.uid} (password reset)`);
} catch (e) {
  if (e.code !== 'auth/user-not-found') throw e;
  user = await auth.createUser({ email: TEST_EMAIL, password: TEST_PASSWORD, emailVerified: true });
  console.log(`✓ Auth user created: ${user.uid}`);
}

// 2. Username must not be claimed by someone else
const unameRef = db.doc(`usernames/${username}`);
const existing = await unameRef.get();
if (existing.exists && existing.data().uid !== user.uid) {
  throw new Error(`Username "${username}" is already claimed by uid ${existing.data().uid}. Pick another TEST_USERNAME.`);
}

// 3. Profile + username claim — exactly the shape from sign-up/page.tsx
const userRef = db.doc(`users/${user.uid}`);
const prior = await userRef.get();

const PROFILE = {
  username,
  email: TEST_EMAIL,
  type: 'advocate',          // only field auth.py require_role() reads
  authProvider: 'password',
  advocateVerified: false,
  createdAt: prior.exists ? prior.data().createdAt : FieldValue.serverTimestamp(),
  updatedAt: FieldValue.serverTimestamp(),
};

const batch = db.batch();
batch.set(userRef, PROFILE);            // full overwrite → no stray fields (rules use hasOnly)
batch.set(unameRef, { uid: user.uid });
await batch.commit();

console.log(`✓ users/${user.uid}  { username: "${username}", type: "advocate", authProvider: "password" }`);
console.log(`✓ usernames/${username}  { uid: "${user.uid}" }`);
console.log('\nNext: npm run test:local -- --headed');
process.exit(0);
