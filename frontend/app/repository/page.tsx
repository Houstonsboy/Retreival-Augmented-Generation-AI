"use client";

import Link from "next/link";
import { Scale, BookOpen, Network, ChevronRight } from "lucide-react";
import { useAuth } from "@/app/context/AuthContext";
import { UserMenu } from "@/app/components/UserMenu";

const cards = [
  {
    href: "/repository/caselaws",
    icon: BookOpen,
    label: "Case Laws",
    description: "Browse, upload, and manage SCC case documents.",
    accent: "text-blue-600 dark:text-blue-400",
    bg: "bg-blue-50 dark:bg-blue-900/20",
    border: "border-blue-100 dark:border-blue-800",
    ring: "hover:ring-blue-200 dark:hover:ring-blue-700",
  },
  {
    href: "/repository/statutes",
    icon: Scale,
    label: "Statutes",
    description: "Search and reference Kenyan statutes and legislation.",
    accent: "text-emerald-600 dark:text-emerald-400",
    bg: "bg-emerald-50 dark:bg-emerald-900/20",
    border: "border-emerald-100 dark:border-emerald-800",
    ring: "hover:ring-emerald-200 dark:hover:ring-emerald-700",
  },
  {
    href: "/knowledge-graph",
    icon: Network,
    label: "Knowledge Graph",
    description: "Explore relationships between cases, statutes, and concepts.",
    accent: "text-violet-600 dark:text-violet-400",
    bg: "bg-violet-50 dark:bg-violet-900/20",
    border: "border-violet-100 dark:border-violet-800",
    ring: "hover:ring-violet-200 dark:hover:ring-violet-700",
  },
];

export default function RepositoryPage() {
  const { profile } = useAuth();

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Top bar */}
      <header className="border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
        <div className="mx-auto max-w-5xl px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gray-900 dark:bg-gray-100">
              <Scale className="h-4 w-4 text-white dark:text-gray-900" />
            </div>
            <span className="text-base font-semibold text-gray-900 dark:text-gray-100">
              LegalAI
            </span>
          </div>
          <UserMenu />
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-12">
        <div className="mb-10">
          <p className="text-xs font-medium uppercase tracking-widest text-gray-400 dark:text-gray-500 mb-1">
            Internal Repository
          </p>
          <h1 className="text-2xl font-semibold text-gray-900 dark:text-gray-100">
            {profile?.username ? `Welcome back, ${profile.username}` : "Repository"}
          </h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Browse legal documents, statutes, and the knowledge graph.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          {cards.map(({ href, icon: Icon, label, description, accent, bg, border, ring }) => (
            <Link
              key={href}
              href={href}
              className={`group relative flex flex-col rounded-xl border ${border} ${bg} p-6 ring-2 ring-transparent transition-all duration-150 ${ring} hover:shadow-sm`}
            >
              <div className={`mb-4 flex h-10 w-10 items-center justify-center rounded-lg bg-white dark:bg-gray-800 shadow-sm`}>
                <Icon className={`h-5 w-5 ${accent}`} />
              </div>

              <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                {label}
              </h2>
              <p className="mt-1 text-xs text-gray-500 dark:text-gray-400 leading-relaxed flex-1">
                {description}
              </p>

              <div className={`mt-4 flex items-center gap-1 text-xs font-medium ${accent}`}>
                Open
                <ChevronRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
              </div>
            </Link>
          ))}
        </div>

        <div className="mt-6">
          <Link
            href="/dashboard"
            className="text-xs text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
          >
            ← Back to Dashboard
          </Link>
        </div>
      </main>
    </div>
  );
}
