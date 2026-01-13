"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
      <div className="mx-auto max-w-7xl px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1">
            <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
              RAG System
            </h1>
          </div>
          <div className="flex items-center gap-2">
            <Link
              href="/"
              className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                pathname === "/"
                  ? "bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400"
                  : "text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
              }`}
            >
              Chat
            </Link>
            <Link
              href="/ingester"
              className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                pathname === "/ingester"
                  ? "bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400"
                  : "text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
              }`}
            >
              Ingester
            </Link>
            <Link
              href="/firacChecker"
              className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                pathname === "/firacChecker"
                  ? "bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400"
                  : "text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
              }`}
            >
              FIRAC
            </Link>
            <Link
              href="/constitution"
              className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                pathname === "/constitution"
                  ? "bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400"
                  : "text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
              }`}
            >
              Constitution
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}

