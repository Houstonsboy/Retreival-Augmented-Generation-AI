"use client";

import { useEffect } from "react";
import { useRouter, usePathname } from "next/navigation";
import { useAuth } from "@/app/context/AuthContext";
import { canAccess, homeRoute, isPublicPath } from "@/app/lib/access";

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const { user, profile, loading, profilePending } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    // profilePending: createLitigantDoc() is in-flight — hold navigation.
    if (loading || profilePending) return;

    const isPublic = isPublicPath(pathname);

    // ── No user ─────────────────────────────────────────────────────────────
    if (!user) {
      if (!isPublic) {
        // Litigant section → litigant sign-in; everything else → advocate sign-in.
        router.replace(
          pathname.startsWith("/litigant") ? "/litigant/sign-in" : "/auth/sign-in"
        );
      }
      return;
    }

    // ── User but no Firestore profile yet ────────────────────────────────────
    if (!profile) {
      if (!pathname.startsWith("/auth/complete-profile")) {
        router.replace("/auth/complete-profile");
      }
      return;
    }

    const type = profile.type;

    // ── /litigant/sign-in ────────────────────────────────────────────────────
    // Litigants: redirect to their area — they're already signed in.
    // Advocates/admins: stay — the page shows a "Continue to evidence tools" notice.
    if (pathname.startsWith("/litigant/sign-in")) {
      if (type === "litigant") router.replace("/litigant/analysis");
      return;
    }

    // ── Landing page (/) ─────────────────────────────────────────────────────
    // Litigants: redirect immediately — there is nothing on the landing page for them.
    // Advocates/admins: stay — they can choose to go to research or evidence tools.
    if (pathname === "/") {
      if (type === "litigant") router.replace("/litigant/analysis");
      return;
    }

    // ── Other public auth pages ───────────────────────────────────────────────
    // (sign-in, sign-up, complete-profile) — redirect all authenticated users home.
    if (isPublic) {
      router.replace(homeRoute(type));
      return;
    }

    // ── Role-gated pages ─────────────────────────────────────────────────────
    // canAccess() encodes the full access map (see app/lib/access.ts).
    if (!canAccess(pathname, type)) {
      router.replace(homeRoute(type));
    }
  }, [user, profile, loading, profilePending, pathname, router]);

  if (loading || profilePending) {
    return (
      <div
        className="flex min-h-screen items-center justify-center"
        style={{ background: "#FAF7F0" }}
      >
        <div
          className="h-8 w-8 animate-spin rounded-full border-4 border-t-transparent"
          style={{ borderColor: "#C9971A", borderTopColor: "transparent" }}
        />
      </div>
    );
  }

  return <>{children}</>;
}
