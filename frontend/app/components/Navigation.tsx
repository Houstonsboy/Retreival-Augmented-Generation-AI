"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard } from "lucide-react";
import { useAuth } from "@/app/context/AuthContext";
import { UserMenu } from "@/app/components/UserMenu";

export default function Navigation() {
  const pathname = usePathname();
  const { profile } = useAuth();
  const isAdmin = profile?.type === "admin";

  const linkClass = (path: string) =>
    `rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
      pathname === path
        ? "bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400"
        : "text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
    }`;

  return (
    <nav className="border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
      <div className="mx-auto max-w-7xl px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
              LegalAI
            </h1>
            <Link
              href="/dashboard"
              className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 hover:text-gray-700 dark:hover:text-gray-300 transition-colors border border-gray-200 dark:border-gray-700"
            >
              <LayoutDashboard className="h-3.5 w-3.5" />
              Dashboard
            </Link>
          </div>

          <div className="flex items-center gap-2">
            <Link href="/" className={linkClass("/")}>Chat</Link>

            {isAdmin && (
              <>
                <Link href="/ingester" className={linkClass("/ingester")}>Ingester</Link>
                <Link href="/firacChecker" className={linkClass("/firacChecker")}>FIRAC</Link>
                <Link href="/constitution" className={linkClass("/constitution")}>Constitution</Link>
              </>
            )}

            <div className="ml-3 pl-3 border-l border-gray-200 dark:border-gray-700">
              <UserMenu />
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}

