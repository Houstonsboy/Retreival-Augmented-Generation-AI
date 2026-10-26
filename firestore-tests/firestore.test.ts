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
import { doc, getDoc, setDoc } from "firebase/firestore";
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

const baseProfile = (email = "user@example.com") => ({
  username: "testuser",
  email,
  type: "advocate" as const,
  authProvider: "password" as const,
  advocateVerified: false,
  createdAt: new Date(),
  updatedAt: new Date(),
});

async function seedUser(uid: string, username: string, type = "advocate") {
  await testEnv.withSecurityRulesDisabled(async (ctx) => {
    const db = ctx.firestore();
    await setDoc(doc(db, "users", uid), {
      ...baseProfile(),
      username,
      type,
    });
    await setDoc(doc(db, "usernames", username), { uid });
  });
}

// ─── Test 1: Can't read another user's document ───────────────────────────────
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

// ─── Test 2: Can't sign up as admin ──────────────────────────────────────────
describe("sign up as admin", () => {
  test("creating a user doc with type=admin is rejected", async () => {
    const db = userCtx("uid_new", "new@example.com").firestore();
    await assertFails(
      setDoc(doc(db, "users", "uid_new"), {
        username: "newuser",
        email: "new@example.com",
        type: "admin", // must be rejected — only "advocate" allowed from client
        authProvider: "password",
        advocateVerified: false,
        createdAt: new Date(),
        updatedAt: new Date(),
      })
    );
  });

  test("creating a user doc with type=advocate succeeds (via batch — tested indirectly)", async () => {
    // Direct setDoc to users/ without a matching usernames/ write is
    // allowed by users rules (no getAfter check on the users side).
    // The batch atomicity constraint lives in the usernames rule.
    const db = userCtx("uid_new", "new@example.com").firestore();
    await assertSucceeds(
      setDoc(doc(db, "users", "uid_new"), {
        username: "newuser",
        email: "new@example.com",
        type: "advocate",
        authProvider: "password",
        advocateVerified: false,
        createdAt: new Date(),
        updatedAt: new Date(),
      })
    );
  });
});

// ─── Test 3: Can't change own type ───────────────────────────────────────────
describe("change own type", () => {
  test("updating type field after creation is rejected", async () => {
    await seedUser("uid_a", "alice");
    const db = userCtx("uid_a").firestore();
    // update (merge) that touches the type field must be denied
    await assertFails(
      setDoc(doc(db, "users", "uid_a"), { type: "admin" }, { merge: true })
    );
  });

  test("updating username is allowed", async () => {
    await seedUser("uid_a", "alice");
    const db = userCtx("uid_a").firestore();
    await assertSucceeds(
      setDoc(
        doc(db, "users", "uid_a"),
        { username: "alice2", updatedAt: new Date() },
        { merge: true }
      )
    );
  });
});

// ─── Test 4: Can't take an existing username ──────────────────────────────────
describe("username uniqueness", () => {
  test("overwriting another user's username entry is rejected", async () => {
    await seedUser("uid_a", "alice"); // alice owns "alice"
    await seedUser("uid_b", "bob");   // uid_b has a valid profile

    const db = userCtx("uid_b", "b@example.com").firestore();
    // "alice" already exists → this is an update, and update is if false
    await assertFails(
      setDoc(doc(db, "usernames", "alice"), { uid: "uid_b" })
    );
  });
});
