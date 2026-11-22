"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  signInWithPopup,
  GoogleAuthProvider,
} from "firebase/auth";
import { serverTimestamp } from "firebase/firestore";
import { auth } from "@/app/lib/firebase";
import { getUserDoc, createLitigantDoc } from "@/app/lib/firestore";
import { Scale, AlertCircle, FileSearch } from "lucide-react";
import { useAuth } from "@/app/context/AuthContext";

// ── Design tokens ──────────────────────────────────────────────────────────
const BG      = "#FAF7F0";
const SURFACE = "#FFFDF8";
const GOLD    = "#C9971A";
const GOLD_DK = "#A87A0C";
const BORDER  = "#ECE6D8";
const INK     = "#151515";
const MUTED   = "#6B6660";
const NR      = "var(--font-newsreader), Georgia, 'Times New Roman', serif";
const MNR     = "var(--font-manrope), system-ui, sans-serif";

function friendlyError(err: unknown): string {
  if (err && typeof err === "object" && "code" in err) {
    const code = (err as { code: string }).code;
    if (code === "auth/popup-closed-by-user" || code === "auth/cancelled-popup-request")
      return "";
    if (code === "auth/too-many-requests")
      return "Too many attempts — please try again later.";
    if (code === "auth/network-request-failed")
      return "Network error. Check your connection and try again.";
  }
  return "Something went wrong. Please try again.";
}

const GoogleIcon = () => (
  <svg viewBox="0 0 24 24" className="h-5 w-5" aria-hidden="true">
    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" />
    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
  </svg>
);

export default function LitigantSignIn() {
  const router = useRouter();
  const { profile, setProfilePending, refreshProfile } = useAuth();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [advocateConflict, setAdvocateConflict] = useState(false);

  // Detect a pre-existing advocate/admin session or a just-signed-in advocate.
  const isAdvocateSession =
    profile?.type === "advocate" || profile?.type === "admin";
  const showAdvocateEntry = isAdvocateSession || advocateConflict;

  const handleGoogle = async () => {
    setError(null);
    setAdvocateConflict(false);
    // Fix 3: pause AuthGuard routing while we create the profile.
    setProfilePending(true);
    setLoading(true);

    try {
      const gProvider = new GoogleAuthProvider();
      gProvider.setCustomParameters({ prompt: "select_account" });
      const result = await signInWithPopup(auth, gProvider);
      const { user } = result;

      // Check for existing profile.
      const existing = await getUserDoc(user.uid).catch(() => null);

      if (existing) {
        if (existing.type === "advocate" || existing.type === "admin") {
          // Don't overwrite — show a clear message.
          setAdvocateConflict(true);
          return;
        }
        // Existing litigant — sync context then navigate.
        await refreshProfile();
        router.replace("/litigant/analysis");
        return;
      }

      // New user — create a litigant profile.
      const ts = { createdAt: serverTimestamp(), updatedAt: serverTimestamp() };
      await createLitigantDoc(user.uid, user.displayName, user.email ?? "", ts);
      // Fix 3: fetch the new profile so AuthGuard sees it before profilePending clears.
      await refreshProfile();
      router.replace("/litigant/analysis");
    } catch (err) {
      const msg = friendlyError(err);
      if (msg) setError(msg);
    } finally {
      // Fix 3: always release the hold so AuthGuard can proceed.
      setProfilePending(false);
      setLoading(false);
    }
  };

  return (
    <div
      className="flex min-h-screen flex-col items-center justify-center px-5"
      style={{ background: BG, fontFamily: MNR }}
    >
      {/* Card */}
      <div
        className="w-full max-w-sm rounded-2xl p-8"
        style={{ background: SURFACE, border: `1px solid ${BORDER}`, boxShadow: "0 8px 32px rgba(60,45,10,0.08)" }}
      >
        {/* Logo */}
        <div className="flex items-center gap-2.5 mb-8">
          <div
            className="h-8 w-8 rounded-xl flex items-center justify-center"
            style={{ background: `linear-gradient(135deg, #7A3A12 0%, ${GOLD} 100%)` }}
          >
            <Scale className="h-4 w-4 text-white" />
          </div>
          <span className="text-sm font-bold tracking-tight" style={{ color: INK }}>
            Kenyan Legal <span style={{ color: GOLD }}>RAG</span>
          </span>
        </div>

        {/* Heading */}
        <p className="text-[10px] font-bold tracking-[.22em] uppercase mb-3" style={{ color: GOLD }}>
          Small Claims Court · Kenya
        </p>
        <h1
          className="text-2xl font-bold leading-snug mb-2"
          style={{ color: INK, fontFamily: NR }}
        >
          Prepare your evidence
        </h1>
        <p className="text-sm leading-relaxed mb-8" style={{ color: MUTED }}>
          Sign in with Google to check your digital evidence — M-Pesa records,
          messages, and written agreements — against the Evidence Act.
        </p>

        {/* Advocate entry notice — shown for signed-in advocates/admins */}
        {showAdvocateEntry && (
          <div
            className="mb-5 rounded-2xl px-5 py-4"
            style={{ background: SURFACE, border: `1.5px solid ${GOLD}44` }}
            role="status"
            aria-label="Advocate access to evidence tools"
          >
            <p
              className="text-[9px] font-bold tracking-[.18em] uppercase mb-1"
              style={{ color: GOLD }}
            >
              Advocate account
            </p>
            <p className="text-sm font-semibold mb-4" style={{ color: INK }}>
              You&apos;re signed in as an advocate.
            </p>
            <Link
              href="/litigant/analysis"
              className="flex items-center justify-center gap-2 w-full rounded-xl py-2.5 text-sm font-bold transition-all hover:brightness-110 focus:outline-none focus:ring-2 focus:ring-brand-gold focus:ring-offset-2"
              style={{ background: `linear-gradient(135deg, ${GOLD_DK}, ${GOLD})`, color: "white" }}
            >
              <FileSearch className="h-4 w-4" aria-hidden="true" />
              Continue to evidence tools
            </Link>
          </div>
        )}

        {/* Error notice */}
        {error && (
          <div
            className="mb-5 flex items-start gap-3 rounded-xl px-4 py-3 text-sm"
            style={{ background: "#FEF2F2", border: "1px solid #FECACA", color: "#B91C1C" }}
            role="alert"
          >
            <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
            {error}
          </div>
        )}

        {/* Google button — hidden while advocate entry notice is visible */}
        <button
          type="button"
          onClick={handleGoogle}
          disabled={loading || showAdvocateEntry}
          className="w-full flex items-center justify-center gap-3 rounded-xl py-3 text-sm font-semibold transition-all hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-brand-gold focus:ring-offset-2"
          style={{
            background: GOLD,
            color: "white",
          }}
          aria-busy={loading}
        >
          {loading ? (
            <span
              className="h-5 w-5 animate-spin rounded-full border-2 border-white border-t-transparent"
              aria-hidden="true"
            />
          ) : (
            <GoogleIcon />
          )}
          {loading ? "Please wait…" : "Continue with Google"}
        </button>

        {/* Privacy note */}
        <p className="mt-6 text-center text-xs" style={{ color: MUTED }}>
          Your files are private to your account.
        </p>
      </div>

      {/* Back link */}
      <p className="mt-6 text-sm" style={{ color: MUTED }}>
        Not representing yourself?{" "}
        <Link href="/" className="font-semibold hover:underline" style={{ color: GOLD_DK }}>
          Back to home
        </Link>
      </p>
    </div>
  );
}
