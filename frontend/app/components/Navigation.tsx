"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/app/context/AuthContext";

export default function Navigation() {
  const pathname = usePathname();
  const router = useRouter();
  const { profile, signOut } = useAuth();
  const isAdmin = profile?.type === "admin";

  const linkClass = (path: string) =>
    `rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
      pathname === path
        ? "bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400"
        : "text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
    }`;

  const handleSignOut = async () => {
    await signOut();
    router.replace("/auth/sign-in");
  };

  return (
    <nav className="border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
      <div className="mx-auto max-w-7xl px-4 py-3">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            LegalAI
          </h1>

          <div className="flex items-center gap-2">
            <Link href="/" className={linkClass("/")}>Chat</Link>

            {isAdmin && (
              <>
                <Link href="/ingester" className={linkClass("/ingester")}>Ingester</Link>
                <Link href="/firacChecker" className={linkClass("/firacChecker")}>FIRAC</Link>
                <Link href="/constitution" className={linkClass("/constitution")}>Constitution</Link>
              </>
            )}

            {profile && (
              <>
                <span className="ml-2 rounded-full bg-gray-100 dark:bg-gray-800 px-3 py-1 text-xs font-medium text-gray-700 dark:text-gray-300">
                  {profile.username}
                  {isAdmin && (
                    <span className="ml-1 text-blue-600 dark:text-blue-400">(admin)</span>
                  )}
                </span>
                <button
                  onClick={handleSignOut}
                  className="rounded-lg px-3 py-1.5 text-sm font-medium text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                >
                  Sign out
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}

