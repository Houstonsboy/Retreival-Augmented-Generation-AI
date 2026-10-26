import {
  doc,
  getDoc,
  writeBatch,
} from "firebase/firestore";
import { db } from "./firebase";

export interface UserProfile {
  username: string;
  email: string;
  type: "advocate" | "admin";
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
