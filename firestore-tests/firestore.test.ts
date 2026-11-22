/**
 * Firestore security rules tests.
 *
 * Run via the project root:
 *   npm run test:rules
 *
 * Or manually:
 *   firebase emulators:exec --only firestore --project legalai-rules-test \
 *     "cd firestore-tests && npm test"
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
  updateDoc,
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

// ── Helpers ───────────────────────────────────────────────────────────────────

function userCtx(uid: string, email = `${uid}@example.com`) {
  return testEnv.authenticatedContext(uid, { email, email_verified: true });
}

const anonCtx = () => testEnv.unauthenticatedContext();

async function seedUser(
  uid: string,
  username: string,
  type: "advocate" | "admin" | "litigant" = "advocate"
) {
  await testEnv.withSecurityRulesDisabled(async (ctx) => {
    const db = ctx.firestore();
    await setDoc(doc(db, "users", uid), {
      username,
      email: `${uid}@example.com`,
      type,
      authProvider: type === "litigant" ? "google" : "password",
      advocateVerified: false,
      createdAt: new Date(),
      updatedAt: new Date(),
    });
    await setDoc(doc(db, "usernames", username), { uid });
  });
}

function advocatePayload(email: string, username: string, authProvider: "password" | "google" = "password") {
  return {
    username,
    email,
    type: "advocate" as const,
    authProvider,
    advocateVerified: false,
    createdAt: serverTimestamp(),
    updatedAt: serverTimestamp(),
  };
}

function litigantPayload(email: string, username: string) {
  return {
    username,
    email,
    type: "litigant" as const,
    authProvider: "google" as const,
    advocateVerified: false,
    createdAt: serverTimestamp(),
    updatedAt: serverTimestamp(),
  };
}

// ── 1. Own profile reads ──────────────────────────────────────────────────────
describe("own profile read", () => {
  test("user can read their own profile", async () => {
    await seedUser("uid_a", "alice");
    const db = userCtx("uid_a").firestore();
    await assertSucceeds(getDoc(doc(db, "users", "uid_a")));
  });

  test("user A cannot read user B's profile", async () => {
    await seedUser("uid_a", "alice");
    await seedUser("uid_b", "bob");
    const db = userCtx("uid_a").firestore();
    await assertFails(getDoc(doc(db, "users", "uid_b")));
  });

  test("unauthenticated user cannot read any profile", async () => {
    await seedUser("uid_a", "alice");
    const db = anonCtx().firestore();
    await assertFails(getDoc(doc(db, "users", "uid_a")));
  });
});

// ── 2. Create: advocate (SHOULD SUCCEED) ─────────────────────────────────────
describe("create advocate profile", () => {
  test("batch with type=advocate, authProvider=password succeeds", async () => {
    const ctx = userCtx("uid_adv", "adv@example.com");
    const db = ctx.firestore();
    const batch = writeBatch(db);
    batch.set(doc(db, "users", "uid_adv"), advocatePayload("adv@example.com", "advuser"));
    batch.set(doc(db, "usernames", "advuser"), { uid: "uid_adv" });
    await assertSucceeds(batch.commit());
  });

  test("batch with type=advocate, authProvider=google succeeds", async () => {
    const ctx = userCtx("uid_gadv", "gadv@example.com");
    const db = ctx.firestore();
    const batch = writeBatch(db);
    batch.set(doc(db, "users", "uid_gadv"), advocatePayload("gadv@example.com", "gadvuser", "google"));
    batch.set(doc(db, "usernames", "gadvuser"), { uid: "uid_gadv" });
    await assertSucceeds(batch.commit());
  });
});

// ── 3. Create: litigant (SHOULD SUCCEED) ─────────────────────────────────────
describe("create litigant profile", () => {
  test("batch with type=litigant, advocateVerified=false succeeds", async () => {
    const ctx = userCtx("uid_lit", "lit@example.com");
    const db = ctx.firestore();
    const batch = writeBatch(db);
    batch.set(doc(db, "users", "uid_lit"), litigantPayload("lit@example.com", "lituser"));
    batch.set(doc(db, "usernames", "lituser"), { uid: "uid_lit" });
    await assertSucceeds(batch.commit());
  });
});

// ── 4. Create: denied cases ───────────────────────────────────────────────────
describe("create profile — denied cases", () => {
  test("type=admin is rejected", async () => {
    const ctx = userCtx("uid_adm", "adm@example.com");
    const db = ctx.firestore();
    const batch = writeBatch(db);
    batch.set(doc(db, "users", "uid_adm"), {
      username: "adminuser",
      email: "adm@example.com",
      type: "admin",
      authProvider: "password",
      advocateVerified: false,
      createdAt: serverTimestamp(),
      updatedAt: serverTimestamp(),
    });
    batch.set(doc(db, "usernames", "adminuser"), { uid: "uid_adm" });
    await assertFails(batch.commit());
  });

  test("advocateVerified=true is rejected", async () => {
    const ctx = userCtx("uid_v", "v@example.com");
    const db = ctx.firestore();
    const batch = writeBatch(db);
    batch.set(doc(db, "users", "uid_v"), {
      username: "vuser",
      email: "v@example.com",
      type: "advocate",
      authProvider: "password",
      advocateVerified: true,   // must be rejected
      createdAt: serverTimestamp(),
      updatedAt: serverTimestamp(),
    });
    batch.set(doc(db, "usernames", "vuser"), { uid: "uid_v" });
    await assertFails(batch.commit());
  });

  test("solo write without matching usernames entry is rejected", async () => {
    const ctx = userCtx("uid_solo", "solo@example.com");
    const db = ctx.firestore();
    await assertFails(
      setDoc(doc(db, "users", "uid_solo"), advocatePayload("solo@example.com", "solouser"))
    );
  });

  test("unauthenticated write to users/* is rejected", async () => {
    const db = anonCtx().firestore();
    await assertFails(
      setDoc(doc(db, "users", "uid_anon"), advocatePayload("anon@example.com", "anonuser"))
    );
  });

  test("writing to another user's uid is rejected", async () => {
    const ctx = userCtx("uid_a", "a@example.com");
    const db = ctx.firestore();
    const batch = writeBatch(db);
    batch.set(doc(db, "users", "uid_b"), advocatePayload("b@example.com", "buser")); // uid mismatch
    batch.set(doc(db, "usernames", "buser"), { uid: "uid_b" });
    await assertFails(batch.commit());
  });
});

// ── 5. Username claim (SHOULD SUCCEED) ───────────────────────────────────────
describe("username claim in same batch", () => {
  test("usernames entry written with the user doc is allowed", async () => {
    const ctx = userCtx("uid_u", "u@example.com");
    const db = ctx.firestore();
    const batch = writeBatch(db);
    batch.set(doc(db, "users", "uid_u"), advocatePayload("u@example.com", "uname"));
    batch.set(doc(db, "usernames", "uname"), { uid: "uid_u" });
    await assertSucceeds(batch.commit());
  });

  test("claiming an existing username is rejected", async () => {
    await seedUser("uid_a", "alice");
    const ctx = userCtx("uid_b", "b@example.com");
    const db = ctx.firestore();
    const batch = writeBatch(db);
    batch.set(doc(db, "users", "uid_b"), advocatePayload("b@example.com", "alice")); // taken
    batch.set(doc(db, "usernames", "alice"), { uid: "uid_b" });
    await assertFails(batch.commit());
  });
});

// ── 6. Update: username change (SHOULD SUCCEED) ───────────────────────────────
describe("update own username", () => {
  test("owner can change username and updatedAt in a batch", async () => {
    await seedUser("uid_a", "alice");
    const db = userCtx("uid_a").firestore();
    const batch = writeBatch(db);
    batch.set(
      doc(db, "users", "uid_a"),
      { username: "alice2", updatedAt: serverTimestamp() },
      { merge: true }
    );
    batch.set(doc(db, "usernames", "alice2"), { uid: "uid_a" });
    await assertSucceeds(batch.commit());
  });
});

// ── 7. Update: frozen fields (SHOULD BE DENIED) ───────────────────────────────
describe("update frozen fields", () => {
  test("owner cannot change type via updateDoc", async () => {
    await seedUser("uid_lit", "lituser", "litigant");
    const db = userCtx("uid_lit").firestore();
    await assertFails(
      updateDoc(doc(db, "users", "uid_lit"), { type: "advocate" })
    );
  });

  test("owner cannot change type via merge setDoc", async () => {
    await seedUser("uid_a", "alice");
    const db = userCtx("uid_a").firestore();
    await assertFails(
      setDoc(doc(db, "users", "uid_a"), { type: "admin" }, { merge: true })
    );
  });

  test("owner cannot change advocateVerified via updateDoc", async () => {
    await seedUser("uid_a", "alice");
    const db = userCtx("uid_a").firestore();
    await assertFails(
      updateDoc(doc(db, "users", "uid_a"), { advocateVerified: true })
    );
  });

  test("owner cannot change advocateVerified via merge setDoc", async () => {
    await seedUser("uid_a", "alice");
    const db = userCtx("uid_a").firestore();
    await assertFails(
      setDoc(doc(db, "users", "uid_a"), { advocateVerified: true }, { merge: true })
    );
  });
});

// ── 8. Username reads ─────────────────────────────────────────────────────────
describe("usernames collection reads", () => {
  test("unauthenticated user can read a username entry (availability check)", async () => {
    await seedUser("uid_a", "alice");
    const db = anonCtx().firestore();
    await assertSucceeds(getDoc(doc(db, "usernames", "alice")));
  });

  test("unauthenticated user can read a non-existent username entry", async () => {
    const db = anonCtx().firestore();
    await assertSucceeds(getDoc(doc(db, "usernames", "nonexistent")));
  });
});
