"use client";

import { useEffect } from "react";
import { useRouter, usePathname } from "next/navigation";
import { useAuth } from "@/app/context/AuthContext";

const PUBLIC_PATHS = ["/auth/sign-in", "/auth/sign-up", "/auth/complete-profile"];
const ADMIN_ONLY_PATHS = ["/ingester", "/firacChecker", "/constitution"];

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const { user, profile, loading } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (loading) return;

    const isPublic = PUBLIC_PATHS.some((p) => pathname.startsWith(p));

    if (!user) {
      if (!isPublic) router.replace("/auth/sign-in");
      return;
    }

    if (!profile) {
      if (!pathname.startsWith("/auth/complete-profile")) {
        router.replace("/auth/complete-profile");
      }
      return;
    }

    if (isPublic) {
      router.replace("/dashboard");
      return;
    }

    if (
      ADMIN_ONLY_PATHS.some((p) => pathname.startsWith(p)) &&
      profile.type !== "admin"
    ) {
      router.replace("/dashboard");
    }
  }, [user, profile, loading, pathname, router]);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-white dark:bg-gray-900">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-600 border-t-transparent" />
      </div>
    );
  }

  return <>{children}</>;
}
