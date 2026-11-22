export const PENDING_USERNAME_KEY = "legalai_pending_username";

import {
  doc,
  getDoc,
  writeBatch,
} from "firebase/firestore";
import { db } from "./firebase";

export interface UserProfile {
  username: string;
  email: string;
  type: "advocate" | "admin" | "litigant";
  authProvider: "password" | "google";
  advocateVerified: boolean;
  createdAt: unknown;
  updatedAt: unknown;
}

export async function getUserDoc(uid: string): Promise<UserProfile | null> {
  const snap = await getDoc(doc(db, "users", uid));
  return snap.exists() ? (snap.data() as UserProfile) : null;
}

export async function isUsernameAvailable(username: string): Promise<boolean> {
  const snap = await getDoc(doc(db, "usernames", username));
  return !snap.exists();
}

// Atomic batch: creates users/{uid} and usernames/{username} together.
// The Firestore rule uses getAfter() on the users doc to enforce atomicity.
export async function createUserDoc(
  uid: string,
  data: Omit<UserProfile, "createdAt" | "updatedAt"> & {
    createdAt: unknown;
    updatedAt: unknown;
  }
): Promise<void> {
  const batch = writeBatch(db);
  batch.set(doc(db, "users", uid), data);
  batch.set(doc(db, "usernames", data.username), { uid });
  await batch.commit();
}

// Derive a safe username from a Google display name / email.
function deriveUsername(displayName: string | null, email: string | null, uid: string): string {
  const base = (displayName || email || uid)
    .toLowerCase()
    .replace(/[^a-z0-9_]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_|_$/g, "")
    .slice(0, 26);
  return base || "user";
}

// Creates a litigant profile. Uses the same atomic batch as createUserDoc.
// If the derived username is taken, appends the last 4 chars of uid.
export async function createLitigantDoc(
  uid: string,
  googleDisplayName: string | null,
  email: string,
  timestamps: { createdAt: unknown; updatedAt: unknown }
): Promise<void> {
  let username = deriveUsername(googleDisplayName, email, uid);

  // Try numeric suffixes _1…_5; fall back to uid-based name if all taken.
  if (!(await isUsernameAvailable(username))) {
    let found = false;
    for (let i = 1; i <= 5; i++) {
      const candidate = `${username}_${i}`.slice(0, 30);
      if (await isUsernameAvailable(candidate)) {
        username = candidate;
        found = true;
        break;
      }
    }
    if (!found) {
      username = `user_${uid.slice(-8)}`.slice(0, 30);
    }
  }

  const batch = writeBatch(db);
  batch.set(doc(db, "users", uid), {
    username,
    email,
    type: "litigant",
    authProvider: "google",
    advocateVerified: false,
    createdAt: timestamps.createdAt,
    updatedAt: timestamps.updatedAt,
  });
  batch.set(doc(db, "usernames", username), { uid });
  await batch.commit();
}
