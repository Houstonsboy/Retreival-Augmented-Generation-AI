"use client";

import Link from "next/link";
import { Scale, ArrowLeft, Construction } from "lucide-react";
import { UserMenu } from "@/app/components/UserMenu";

export default function StatutesPage() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <header className="border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
        <div className="mx-auto max-w-5xl px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gray-900 dark:bg-gray-100">
              <Scale className="h-4 w-4 text-white dark:text-gray-900" />
            </div>
            <span className="text-base font-semibold text-gray-900 dark:text-gray-100">LegalAI</span>
          </div>
          <UserMenu />
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-10">
        <Link
          href="/repository"
          className="flex items-center gap-1 text-xs text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300 transition-colors mb-8"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Repository
        </Link>

        <div className="flex flex-col items-center justify-center py-24 text-center">
          <Construction className="h-10 w-10 text-gray-300 dark:text-gray-600 mb-4" />
          <h2 className="text-lg font-semibold text-gray-700 dark:text-gray-300">Statutes — Coming Soon</h2>
          <p className="mt-2 text-sm text-gray-400 dark:text-gray-500">This section is being built.</p>
        </div>
      </main>
    </div>
  );
}
