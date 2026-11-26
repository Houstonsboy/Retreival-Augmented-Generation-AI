"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  signInWithEmailAndPassword,
  signInWithPopup,
  GoogleAuthProvider,
  sendPasswordResetEmail,
} from "firebase/auth";
import { Scale, CheckCircle2 } from "lucide-react";
import { auth } from "@/app/lib/firebase";
import { getUserDoc } from "@/app/lib/firestore";

// ── Design tokens ──────────────────────────────────────────────────────────
const BG      = "#FAF7F0";
const GOLD    = "#C9971A";
const GOLD_DK = "#A87A0C";
const BORDER  = "#ECE6D8";
const INK     = "#151515";
const MUTED   = "#6B6660";
const DARK    = "#1C1409";
const MNR     = "var(--font-manrope), system-ui, sans-serif";
const NR      = "var(--font-newsreader), Georgia, 'Times New Roman', serif";

function friendlyError(err: unknown): string {
  if (err && typeof err === "object" && "code" in err) {
    const code = (err as { code: string }).code;
    if (code === "auth/invalid-credential" || code === "auth/wrong-password")
      return "Invalid email or password.";
    if (code === "auth/user-not-found")
      return "No account found with this email.";
    if (code === "auth/too-many-requests")
      return "Too many attempts — please try again later.";
    if (code === "auth/popup-closed-by-user" || code === "auth/cancelled-popup-request")
      return "";
    if (code === "auth/unauthorized-domain")
      return "This domain isn't authorised for sign-in. Add it in Firebase → Authentication → Authorized domains.";
  }
  return "An error occurred. Please try again.";
}

const GoogleIcon = () => (
  <svg viewBox="0 0 24 24" className="h-4 w-4 shrink-0">
    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" />
    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
  </svg>
);

const bullets = [
  "Answers cite Kenyan case law and statutes",
  "Evidence checked against Section 106B of the Evidence Act",
  "Signed certificate drafts, ready to file",
];

export default function SignIn() {
  const router = useRouter();
  const [email, setEmail]       = useState("");
  const [password, setPassword] = useState("");
  const [error, setError]       = useState<string | null>(null);
  const [resetSent, setResetSent] = useState(false);
  const [loading, setLoading]   = useState(false);

  const handleEmailSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await signInWithEmailAndPassword(auth, email, password);
      router.replace("/dashboard");
    } catch (err) {
      setError(friendlyError(err));
      setLoading(false);
    }
  };

  const handleGoogle = async () => {
    setError(null);
    setLoading(true);
    try {
      const gProvider = new GoogleAuthProvider();
      gProvider.setCustomParameters({ prompt: "select_account" });
      const result = await signInWithPopup(auth, gProvider);
      const profile = await getUserDoc(result.user.uid).catch(() => null);
      router.replace(profile ? "/dashboard" : "/auth/complete-profile");
    } catch (err) {
      const msg = friendlyError(err);
      if (msg) setError(msg);
      setLoading(false);
    }
  };

  const handleReset = async () => {
    if (!email) { setError("Enter your email address first."); return; }
    setLoading(true);
    try {
      await sendPasswordResetEmail(auth, email);
      setResetSent(true);
      setError(null);
    } catch (err) {
      setError(friendlyError(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen" style={{ fontFamily: MNR }}>
      {/* ── Left panel — dark brand ──────────────────────────────────── */}
      <div
        className="hidden lg:flex flex-col justify-between w-[46%] shrink-0 p-10"
        style={{ background: DARK }}
      >
        {/* Logo */}
        <div className="flex items-center gap-2.5">
          <div
            className="h-8 w-8 rounded-xl flex items-center justify-center shrink-0"
            style={{ background: `linear-gradient(135deg, #7A3A12 0%, ${GOLD} 100%)` }}
          >
            <Scale className="h-4 w-4 text-white" />
          </div>
          <span className="text-sm font-bold tracking-tight text-white">
            Kenyan Legal{" "}
            <span style={{ color: GOLD }}>RAG</span>
          </span>
        </div>

        {/* Hero text */}
        <div className="flex-1 flex flex-col justify-center max-w-sm py-12">
          <p
            className="text-[10px] font-bold tracking-[.28em] uppercase mb-6"
            style={{ color: GOLD }}
          >
            Small Claims Court · Kenya
          </p>
          <h2
            className="text-3xl font-bold leading-snug mb-5 text-white"
            style={{ fontFamily: NR, fontStyle: "italic" }}
          >
            Legal research and evidence preparation, grounded in Kenyan law.
          </h2>
          <p className="text-sm leading-relaxed mb-8" style={{ color: "#A89E8E" }}>
            Ask questions answered from case law, statutes and the Constitution, and
            prepare WhatsApp and M-Pesa evidence for court.
          </p>

          <ul className="space-y-3.5">
            {bullets.map((b) => (
              <li key={b} className="flex items-start gap-3">
                <CheckCircle2 className="h-4 w-4 mt-0.5 shrink-0" style={{ color: GOLD }} />
                <span className="text-sm leading-relaxed" style={{ color: "#D4C9B0" }}>
                  {b}
                </span>
              </li>
            ))}
          </ul>
        </div>

        {/* Footer */}
        <p className="text-xs" style={{ color: "#6B6056" }}>
          Automated analysis — not legal advice
        </p>
      </div>

      {/* ── Right panel — form ───────────────────────────────────────── */}
      <div
        className="flex flex-1 flex-col items-center justify-center px-6 py-12"
        style={{ background: BG }}
      >
        {/* Mobile logo */}
        <div className="flex lg:hidden items-center gap-2 mb-8">
          <div
            className="h-7 w-7 rounded-lg flex items-center justify-center"
            style={{ background: `linear-gradient(135deg, #7A3A12 0%, ${GOLD} 100%)` }}
          >
            <Scale className="h-3.5 w-3.5 text-white" />
          </div>
          <span className="text-sm font-bold tracking-tight" style={{ color: INK }}>
            Kenyan Legal <span style={{ color: GOLD }}>RAG</span>
          </span>
        </div>

        <div className="w-full max-w-sm">
          <h1
            className="text-2xl font-bold mb-1"
            style={{ color: INK }}
          >
            Sign in
          </h1>
          <p className="text-sm mb-7" style={{ color: MUTED }}>
            Welcome back. Sign in to continue to your workspace.
          </p>

          {/* Error / success banners */}
          {error && (
            <div
              className="mb-4 rounded-xl border px-4 py-3 text-sm"
              style={{ borderColor: "#F5C6C6", background: "#FEF2F2", color: "#B91C1C" }}
            >
              {error}
            </div>
          )}
          {resetSent && (
            <div
              className="mb-4 rounded-xl border px-4 py-3 text-sm"
              style={{ borderColor: "#A7F3D0", background: "#F0FDF4", color: "#065F46" }}
            >
              Password reset email sent.
            </div>
          )}

          <form onSubmit={handleEmailSignIn} className="space-y-4">
            {/* Email */}
            <div>
              <label
                className="block text-sm font-medium mb-1.5"
                style={{ color: INK }}
              >
                Email
              </label>
              <input
                type="email"
                required
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full rounded-xl px-3.5 py-2.5 text-sm focus:outline-none focus:ring-2"
                style={{
                  border: `1px solid ${BORDER}`,
                  background: "#FFFDF8",
                  color: INK,
                }}
              />
            </div>

            {/* Password */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label
                  className="text-sm font-medium"
                  style={{ color: INK }}
                >
                  Password
                </label>
                <button
                  type="button"
                  onClick={handleReset}
                  disabled={loading}
                  className="text-xs font-medium hover:underline disabled:opacity-50"
                  style={{ color: GOLD_DK }}
                >
                  Forgot password?
                </button>
              </div>
              <input
                type="password"
                required
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full rounded-xl px-3.5 py-2.5 text-sm focus:outline-none focus:ring-2"
                style={{
                  border: `1px solid ${BORDER}`,
                  background: "#FFFDF8",
                  color: INK,
                }}
              />
            </div>

            {/* Submit */}
            <button
              type="submit"
              disabled={loading}
              className="w-full rounded-xl py-2.5 text-sm font-semibold text-white transition-opacity disabled:opacity-60"
              style={{ background: GOLD }}
            >
              {loading ? "Please wait…" : "Sign in"}
            </button>
          </form>

          {/* Divider */}
          <div className="my-5 flex items-center gap-3">
            <div className="flex-1 border-t" style={{ borderColor: BORDER }} />
            <span className="text-xs" style={{ color: MUTED }}>or</span>
            <div className="flex-1 border-t" style={{ borderColor: BORDER }} />
          </div>

          {/* Google */}
          <button
            type="button"
            onClick={handleGoogle}
            disabled={loading}
            className="w-full rounded-xl py-2.5 text-sm font-medium flex items-center justify-center gap-2.5 transition-colors disabled:opacity-60"
            style={{
              border: `1px solid ${BORDER}`,
              background: "#FFFDF8",
              color: INK,
            }}
          >
            <GoogleIcon />
            Continue with Google
          </button>

          <p className="mt-6 text-center text-sm" style={{ color: MUTED }}>
            Don&apos;t have an account?{" "}
            <Link
              href="/auth/sign-up"
              className="font-semibold hover:underline"
              style={{ color: GOLD_DK }}
            >
              Create one
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
