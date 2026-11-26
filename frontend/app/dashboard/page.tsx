"use client";

import { useRouter } from "next/navigation";
import { MessageSquare, FolderOpen, ArrowRight, Scale, FileSearch } from "lucide-react";
import { useAuth } from "@/app/context/AuthContext";
import { UserMenu } from "@/app/components/UserMenu";

export default function Dashboard() {
  const router = useRouter();
  const { profile } = useAuth();

  const tools = [
    {
      icon: MessageSquare,
      label: "Chat",
      description: "Ask legal questions and get analysis grounded in Kenyan case law and statutes.",
      href: "/research",
      cta: "Open chat",
      accent: "from-blue-50 to-indigo-50 dark:from-blue-950/30 dark:to-indigo-950/30",
      border: "border-blue-100 dark:border-blue-900/50",
      iconBg: "bg-blue-100 dark:bg-blue-900/50",
      iconColor: "text-blue-600 dark:text-blue-400",
      ctaColor: "text-blue-600 dark:text-blue-400 group-hover:text-blue-700 dark:group-hover:text-blue-300",
      arrowColor: "text-blue-400 dark:text-blue-500 group-hover:text-blue-600 dark:group-hover:text-blue-300",
    },
    {
      icon: FolderOpen,
      label: "Internal Repository",
      description: "Browse and manage uploaded case documents, statutes, and legal references.",
      href: "/repository",
      cta: "Browse files",
      accent: "from-amber-50 to-orange-50 dark:from-amber-950/30 dark:to-orange-950/30",
      border: "border-amber-100 dark:border-amber-900/50",
      iconBg: "bg-amber-100 dark:bg-amber-900/50",
      iconColor: "text-amber-600 dark:text-amber-400",
      ctaColor: "text-amber-600 dark:text-amber-400 group-hover:text-amber-700 dark:group-hover:text-amber-300",
      arrowColor: "text-amber-400 dark:text-amber-500 group-hover:text-amber-600 dark:group-hover:text-amber-300",
    },
    {
      icon: FileSearch,
      label: "Evidence Tools",
      description: "Prepare digital evidence for Small Claims Court — M-Pesa records, messages, and agreements.",
      href: "/litigant/analysis",
      cta: "Open evidence tools",
      // brand gold palette (--brand-gold #C9971A / --brand-highlight #F6E7B8 / --brand-border #ECE6D8)
      accent: "from-[#FAF7F0] to-[#FFFDF8]",
      border: "border-[#ECE6D8]",
      iconBg: "bg-[#F6E7B8]",
      iconColor: "text-[#C9971A]",
      ctaColor: "text-[#A87A0C] group-hover:text-[#C9971A]",
      arrowColor: "text-[#C9971A] group-hover:text-[#A87A0C]",
    },
  ];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 flex flex-col">
      {/* Top bar */}
      <header className="border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
        <div className="max-w-5xl mx-auto px-5 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="h-7 w-7 rounded-lg bg-blue-600 flex items-center justify-center">
              <Scale className="h-4 w-4 text-white" />
            </div>
            <span className="font-semibold text-gray-900 dark:text-gray-100 tracking-tight">
              LegalAI
            </span>
          </div>
          <UserMenu />
        </div>
      </header>

      {/* Body */}
      <main className="flex-1 flex flex-col items-center justify-center px-5 py-16">
        {/* Greeting */}
        <div className="text-center mb-12">
          <p className="text-xs font-semibold uppercase tracking-widest text-gray-400 dark:text-gray-500 mb-2">
            Welcome back
          </p>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100 tracking-tight">
            {profile?.username ?? "Counsellor"}
          </h1>
          <p className="mt-2 text-sm text-gray-500 dark:text-gray-400 max-w-xs">
            What would you like to do today?
          </p>
        </div>

        {/* Navigation cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full max-w-2xl">
          {tools.map((tool) => {
            const Icon = tool.icon;
            return (
              <button
                key={tool.label}
                onClick={() => router.push(tool.href)}
                className={`group relative flex flex-col items-start text-left p-6 rounded-2xl border bg-gradient-to-br ${tool.accent} ${tool.border} transition-all duration-200 hover:shadow-md hover:-translate-y-0.5 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 dark:focus:ring-offset-gray-950`}
              >
                {/* Icon */}
                <div className={`h-10 w-10 rounded-xl ${tool.iconBg} flex items-center justify-center mb-4`}>
                  <Icon className={`h-5 w-5 ${tool.iconColor}`} />
                </div>

                {/* Text */}
                <h2 className="text-base font-semibold text-gray-900 dark:text-gray-100 mb-1">
                  {tool.label}
                </h2>
                <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed flex-1">
                  {tool.description}
                </p>

                {/* CTA row */}
                <div className={`mt-5 flex items-center gap-1.5 text-sm font-medium ${tool.ctaColor} transition-colors`}>
                  {tool.cta}
                  <ArrowRight className={`h-4 w-4 transition-transform duration-200 group-hover:translate-x-1 ${tool.arrowColor}`} />
                </div>
              </button>
            );
          })}
        </div>
      </main>
    </div>
  );
}
