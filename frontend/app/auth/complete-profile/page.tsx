"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { serverTimestamp } from "firebase/firestore";
import { auth } from "@/app/lib/firebase";
import { createUserDoc, isUsernameAvailable, PENDING_USERNAME_KEY } from "@/app/lib/firestore";

function validateUsername(v: string): string | null {
  if (v.length < 3) return "At least 3 characters.";
  if (v.length > 30) return "Max 30 characters.";
  if (!/^[a-z0-9_]+$/.test(v)) return "Lowercase, numbers, underscore only.";
  return null;
}

export default function CompleteProfile() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [usernameError, setUsernameError] = useState<string | null>(null);
  const [usernameOk, setUsernameOk] = useState(false);
  const [checkingUsername, setCheckingUsername] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Pre-fill username saved before the Google redirect
  useEffect(() => {
    const pending = sessionStorage.getItem(PENDING_USERNAME_KEY);
    if (pending) setUsername(pending);
  }, []);

  const checkUsername = useCallback(async (value: string) => {
    const validErr = validateUsername(value);
    if (validErr) { setUsernameError(validErr); setUsernameOk(false); return; }
    setCheckingUsername(true);
    const available = await isUsernameAvailable(value);
    setCheckingUsername(false);
    if (available) { setUsernameError(null); setUsernameOk(true); }
    else { setUsernameError("Username already taken."); setUsernameOk(false); }
  }, []);

  useEffect(() => {
    setUsernameOk(false); setUsernameError(null);
    if (!username) return;
    const t = setTimeout(() => checkUsername(username), 500);
    return () => clearTimeout(t);
  }, [username, checkUsername]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!usernameOk) return;

    const user = auth.currentUser;
    if (!user) { setError("Session expired — please sign in again."); return; }

    setError(null); setLoading(true);
    try {
      await createUserDoc(user.uid, {
        username,
        email: user.email!,
        type: "advocate",
        authProvider: "google",
        advocateVerified: false,
        createdAt: serverTimestamp(),
        updatedAt: serverTimestamp(),
      });
      sessionStorage.removeItem(PENDING_USERNAME_KEY);
      router.replace("/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save profile.");
      setLoading(false);
    }
  };

  const usernameInputClass = `w-full rounded-lg border px-3 py-2 text-sm pr-8 focus:outline-none focus:ring-2 ${
    usernameOk ? "border-green-400 dark:border-green-600 focus:ring-green-500"
    : usernameError ? "border-red-400 dark:border-red-600 focus:ring-red-500"
    : "border-gray-300 dark:border-gray-600 focus:ring-blue-500"
  } bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100`;

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 dark:bg-gray-900 px-4">
      <div className="w-full max-w-sm">
        <h1 className="mb-2 text-center text-2xl font-semibold text-gray-900 dark:text-gray-100">
          Almost there
        </h1>
        <p className="mb-6 text-center text-sm text-gray-500 dark:text-gray-400">
          Choose a username to complete your profile.
        </p>

        {error && (
          <div className="mb-4 rounded-lg border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20 px-4 py-3 text-sm text-red-800 dark:text-red-200">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Username</label>
            <div className="relative">
              <input type="text" value={username}
                onChange={(e) => setUsername(e.target.value.toLowerCase())}
                placeholder="e.g. jane_smith" className={usernameInputClass} />
              {checkingUsername && (
                <div className="absolute right-2 top-1/2 -translate-y-1/2 h-4 w-4 animate-spin rounded-full border-2 border-blue-500 border-t-transparent" />
              )}
              {!checkingUsername && usernameOk && (
                <span className="absolute right-2 top-1/2 -translate-y-1/2 text-green-500 text-xs">✓</span>
              )}
            </div>
            {usernameError && <p className="mt-1 text-xs text-red-600 dark:text-red-400">{usernameError}</p>}
            {!usernameError && usernameOk && <p className="mt-1 text-xs text-green-600 dark:text-green-400">Username available</p>}
            <p className="mt-1 text-xs text-gray-500">3-30 chars · lowercase, numbers, underscore only</p>
          </div>

          <button type="submit" disabled={loading || !usernameOk}
            className="w-full rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors">
            {loading ? "Saving…" : "Save username"}
          </button>
        </form>
      </div>
    </div>
  );
}
