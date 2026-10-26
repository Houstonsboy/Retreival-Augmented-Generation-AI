/**
 * Firestore security rules tests.
 *
 * Prerequisites:
 *   firebase emulators:start --only firestore
 *   (emulator must be on port 8080 — see firebase.json)
 *
 * Run:
 *   cd firestore-tests && npm install && npm test
 */
import {
  assertFails,
  assertSucceeds,
  initializeTestEnvironment,
  RulesTestEnvironment,
} from "@firebase/rules-unit-testing";
import {
  doc,
  getDoc,
  setDoc,
  writeBatch,
  serverTimestamp,
} from "firebase/firestore";
import { readFileSync } from "fs";
import { resolve } from "path";

let testEnv: RulesTestEnvironment;

beforeAll(async () => {
  testEnv = await initializeTestEnvironment({
    projectId: "legalai-rules-test",
    firestore: {
      rules: readFileSync(resolve(__dirname, "../firestore.rules"), "utf8"),
      host: "127.0.0.1",
      port: 8080,
    },
  });
});

afterAll(async () => {
  await testEnv.cleanup();
});

afterEach(async () => {
  await testEnv.clearFirestore();
});

function userCtx(uid: string, email = "user@example.com") {
  return testEnv.authenticatedContext(uid, { email, email_verified: true });
}

// Seed bypasses rules — uses new Date() because rules don't apply here.
async function seedUser(uid: string, username: string, type = "advocate") {
  await testEnv.withSecurityRulesDisabled(async (ctx) => {
    const db = ctx.firestore();
    await setDoc(doc(db, "users", uid), {
      username,
      email: `${uid}@example.com`,
      type,
      authProvider: "password",
      advocateVerified: false,
      createdAt: new Date(),
      updatedAt: new Date(),
    });
    await setDoc(doc(db, "usernames", username), { uid });
  });
}

// Valid batch payload for a new password-auth user going through rules.
function passwordProfile(email: string, username: string) {
  return {
    username,
    email,
    type: "advocate" as const,
    authProvider: "password" as const,
    advocateVerified: false,
    createdAt: serverTimestamp(),
    updatedAt: serverTimestamp(),
  };
}

// Valid batch payload for a new Google-auth user going through rules.
function googleProfile(email: string, username: string) {
  return {
    username,
    email,
    type: "advocate" as const,
    authProvider: "google" as const,
    advocateVerified: false,
    createdAt: serverTimestamp(),
    updatedAt: serverTimestamp(),
  };
}

// ─── 1. Cross-user read ──────────────────────────────────────────────────────
describe("cross-user read", () => {
  test("user A cannot read user B's profile", async () => {
    await seedUser("uid_a", "alice");
    await seedUser("uid_b", "bob");
    const db = userCtx("uid_a").firestore();
    await assertFails(getDoc(doc(db, "users", "uid_b")));
  });

  test("user can read their own profile", async () => {
    await seedUser("uid_a", "alice");
    const db = userCtx("uid_a").firestore();
    await assertSucceeds(getDoc(doc(db, "users", "uid_a")));
  });
});

// ─── 2. Can't sign up as admin ───────────────────────────────────────────────
describe("sign up as admin", () => {
  test("batch with type=admin is rejected", async () => {
    const ctx = userCtx("uid_new", "new@example.com");
    const db = ctx.firestore();
    const batch = writeBatch(db);
    batch.set(doc(db, "users", "uid_new"), {
      username: "newuser",
      email: "new@example.com",
      type: "admin",              // must be rejected
      authProvider: "password",
      advocateVerified: false,
      createdAt: serverTimestamp(),
      updatedAt: serverTimestamp(),
    });
    batch.set(doc(db, "usernames", "newuser"), { uid: "uid_new" });
    await assertFails(batch.commit());
  });

  test("batch with type=advocate succeeds (password auth)", async () => {
    const ctx = userCtx("uid_new", "new@example.com");
    const db = ctx.firestore();
    const batch = writeBatch(db);
    batch.set(doc(db, "users", "uid_new"), passwordProfile("new@example.com", "newuser"));
    batch.set(doc(db, "usernames", "newuser"), { uid: "uid_new" });
    await assertSucceeds(batch.commit());
  });
});

// ─── 3. Google user profile creation ─────────────────────────────────────────
describe("Google user sign-up", () => {
  test("batch with authProvider=google succeeds", async () => {
    const ctx = userCtx("uid_google", "guser@example.com");
    const db = ctx.firestore();
    const batch = writeBatch(db);
    batch.set(doc(db, "users", "uid_google"), googleProfile("guser@example.com", "guser"));
    batch.set(doc(db, "usernames", "guser"), { uid: "uid_google" });
    await assertSucceeds(batch.commit());
  });

  test("authProvider=google.com is rejected", async () => {
    const ctx = userCtx("uid_google", "guser@example.com");
    const db = ctx.firestore();
    const batch = writeBatch(db);
    batch.set(doc(db, "users", "uid_google"), {
      username: "guser",
      email: "guser@example.com",
      type: "advocate",
      authProvider: "google.com",   // Firebase provider ID — must be rejected
      advocateVerified: false,
      createdAt: serverTimestamp(),
      updatedAt: serverTimestamp(),
    });
    batch.set(doc(db, "usernames", "guser"), { uid: "uid_google" });
    await assertFails(batch.commit());
  });

  test("users write without matching usernames write is rejected", async () => {
    const ctx = userCtx("uid_google", "guser@example.com");
    const db = ctx.firestore();
    // Standalone setDoc — existsAfter(usernamePath) returns false
    await assertFails(
      setDoc(doc(db, "users", "uid_google"), googleProfile("guser@example.com", "guser"))
    );
  });

  test("usernames write without matching users write is rejected", async () => {
    const ctx = userCtx("uid_google", "guser@example.com");
    const db = ctx.firestore();
    // Standalone setDoc — existsAfter(userPath) returns false
    await assertFails(
      setDoc(doc(db, "usernames", "guser"), { uid: "uid_google" })
    );
  });
});

// ─── 4. Can't change own type ────────────────────────────────────────────────
describe("change own type", () => {
  test("merging type=admin is rejected", async () => {
    await seedUser("uid_a", "alice");
    const db = userCtx("uid_a").firestore();
    await assertFails(
      setDoc(doc(db, "users", "uid_a"), { type: "admin" }, { merge: true })
    );
  });

  test("updating username via batch is allowed", async () => {
    await seedUser("uid_a", "alice");
    const db = userCtx("uid_a").firestore();
    const batch = writeBatch(db);
    // users update: only username + updatedAt change; new usernames entry in same batch
    batch.set(
      doc(db, "users", "uid_a"),
      { username: "alice2", updatedAt: serverTimestamp() },
      { merge: true }
    );
    batch.set(doc(db, "usernames", "alice2"), { uid: "uid_a" });
    await assertSucceeds(batch.commit());
  });
});

// ─── 5. Can't take an existing username ──────────────────────────────────────
describe("username uniqueness", () => {
  test("overwriting another user's username entry is rejected", async () => {
    await seedUser("uid_a", "alice");
    await seedUser("uid_b", "bob");
    const db = userCtx("uid_b", "b@example.com").firestore();
    // "alice" already exists — update rule is `if false`
    await assertFails(
      setDoc(doc(db, "usernames", "alice"), { uid: "uid_b" })
    );
  });
});
