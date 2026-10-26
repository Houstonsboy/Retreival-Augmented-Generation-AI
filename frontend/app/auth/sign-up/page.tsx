"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  createUserWithEmailAndPassword,
  signInWithPopup,
  GoogleAuthProvider,
} from "firebase/auth";
import { serverTimestamp } from "firebase/firestore";
import { auth } from "@/app/lib/firebase";
import { createUserDoc, isUsernameAvailable } from "@/app/lib/firestore";

function validateUsername(v: string): string | null {
  if (v.length < 3) return "At least 3 characters.";
  if (v.length > 30) return "Max 30 characters.";
  if (!/^[a-z0-9_]+$/.test(v)) return "Lowercase, numbers, underscore only.";
  return null;
}

function friendlyError(err: unknown): string {
  if (err && typeof err === "object" && "code" in err) {
    const code = (err as { code: string }).code;
    if (code === "auth/email-already-in-use")
      return "An account with this email already exists.";
    if (code === "auth/weak-password") return "Password is too weak.";
    if (code === "auth/invalid-email") return "Invalid email address.";
    if (code === "auth/popup-closed-by-user") return "Sign-in cancelled.";
  }
  return "An error occurred. Please try again.";
}

const GoogleIcon = () => (
  <svg viewBox="0 0 24 24" className="h-4 w-4">
    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" />
    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
  </svg>
);

export default function SignUp() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [usernameError, setUsernameError] = useState<string | null>(null);
  const [usernameOk, setUsernameOk] = useState(false);
  const [checkingUsername, setCheckingUsername] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const checkUsername = useCallback(async (value: string) => {
    const validErr = validateUsername(value);
    if (validErr) {
      setUsernameError(validErr);
      setUsernameOk(false);
      return;
    }
    setCheckingUsername(true);
    const available = await isUsernameAvailable(value);
    setCheckingUsername(false);
    if (available) {
      setUsernameError(null);
      setUsernameOk(true);
    } else {
      setUsernameError("Username already taken.");
      setUsernameOk(false);
    }
  }, []);

  useEffect(() => {
    setUsernameOk(false);
    setUsernameError(null);
    if (!username) return;
    const t = setTimeout(() => checkUsername(username), 500);
    return () => clearTimeout(t);
  }, [username, checkUsername]);

  const handleEmailSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!usernameOk) { setError("Fix username before continuing."); return; }
    setError(null);
    setLoading(true);
    try {
      const cred = await createUserWithEmailAndPassword(auth, email, password);
      await createUserDoc(cred.user.uid, {
        username,
        email,
        type: "advocate",
        authProvider: "password",
        advocateVerified: false,
        createdAt: serverTimestamp(),
        updatedAt: serverTimestamp(),
      });
      router.replace("/");
    } catch (err) {
      setError(friendlyError(err));
    } finally {
      setLoading(false);
    }
  };

  const handleGoogle = async () => {
    if (!usernameOk) { setError("Fix username before continuing."); return; }
    setError(null);
    setLoading(true);
    try {
      const result = await signInWithPopup(auth, new GoogleAuthProvider());
      await createUserDoc(result.user.uid, {
        username,
        email: result.user.email!,
        type: "advocate",
        authProvider: "google",
        advocateVerified: false,
        createdAt: serverTimestamp(),
        updatedAt: serverTimestamp(),
      });
      router.replace("/");
    } catch (err) {
      setError(friendlyError(err));
    } finally {
      setLoading(false);
    }
  };

  const usernameInputClass = `w-full rounded-lg border px-3 py-2 text-sm pr-8 focus:outline-none focus:ring-2 ${
    usernameOk
      ? "border-green-400 dark:border-green-600 focus:ring-green-500"
      : usernameError
      ? "border-red-400 dark:border-red-600 focus:ring-red-500"
      : "border-gray-300 dark:border-gray-600 focus:ring-blue-500"
  } bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100`;

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 dark:bg-gray-900 px-4">
      <div className="w-full max-w-sm">
        <h1 className="mb-6 text-center text-2xl font-semibold text-gray-900 dark:text-gray-100">
          Create your account
        </h1>

        {error && (
          <div className="mb-4 rounded-lg border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20 px-4 py-3 text-sm text-red-800 dark:text-red-200">
            {error}
          </div>
        )}

        {/* Username (outside the form so it applies to both sign-up flows) */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Username
          </label>
          <div className="relative">
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value.toLowerCase())}
              placeholder="e.g. jane_smith"
              className={usernameInputClass}
            />
            {checkingUsername && (
              <div className="absolute right-2 top-1/2 -translate-y-1/2 h-4 w-4 animate-spin rounded-full border-2 border-blue-500 border-t-transparent" />
            )}
            {!checkingUsername && usernameOk && (
              <span className="absolute right-2 top-1/2 -translate-y-1/2 text-green-500 text-xs">
                ✓
              </span>
            )}
          </div>
          {usernameError && (
            <p className="mt-1 text-xs text-red-600 dark:text-red-400">
              {usernameError}
            </p>
          )}
          {!usernameError && usernameOk && (
            <p className="mt-1 text-xs text-green-600 dark:text-green-400">
              Username available
            </p>
          )}
        </div>

        <form onSubmit={handleEmailSignUp} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Email
            </label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Password
            </label>
            <input
              type="password"
              required
              minLength={8}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <p className="mt-1 text-xs text-gray-500">Minimum 8 characters</p>
          </div>

          <button
            type="submit"
            disabled={loading || !usernameOk}
            className="w-full rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? "Creating account…" : "Sign up with email"}
          </button>
        </form>

        <div className="my-4 flex items-center gap-2">
          <div className="flex-1 border-t border-gray-200 dark:border-gray-700" />
          <span className="text-xs text-gray-500">or</span>
          <div className="flex-1 border-t border-gray-200 dark:border-gray-700" />
        </div>

        <button
          type="button"
          onClick={handleGoogle}
          disabled={loading || !usernameOk}
          className="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50 flex items-center justify-center gap-2 transition-colors"
        >
          <GoogleIcon />
          Continue with Google
        </button>

        <p className="mt-6 text-center text-sm text-gray-600 dark:text-gray-400">
          Already have an account?{" "}
          <Link
            href="/auth/sign-in"
            className="text-blue-600 dark:text-blue-400 hover:underline"
          >
            Sign in
          </Link>
        </p>
      </div>
    </div>
  );
}
