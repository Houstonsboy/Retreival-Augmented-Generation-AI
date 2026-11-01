"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import { LogOut, ChevronDown } from "lucide-react";
import { useAuth } from "@/app/context/AuthContext";

export function UserMenu() {
  const { profile, user, signOut } = useAuth();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  if (!profile) return null;

  const initials = profile.username.slice(0, 2).toUpperCase();
  const isAdmin = profile.type === "admin";

  const handleSignOut = async () => {
    setConfirming(false);
    await signOut();
    router.replace("/auth/sign-in");
  };

  return (
    <>
      <div className="relative" ref={menuRef}>
        {/* Avatar button */}
        <button
          onClick={() => setOpen((v) => !v)}
          className="flex items-center gap-2 rounded-full pl-1 pr-2 py-1 hover:bg-amber-50 transition-colors focus:outline-none focus:ring-2 focus:ring-amber-400 focus:ring-offset-1"
          aria-label="User menu"
        >
          <div className="h-8 w-8 rounded-full flex items-center justify-center text-white text-sm font-bold select-none" style={{ background: "#151515" }}>
            {initials}
          </div>
          <span className="hidden sm:block text-sm font-medium text-gray-700 max-w-[120px] truncate">
            {profile.username}
          </span>
          {isAdmin && (
            <span className="hidden sm:block text-[10px] font-bold tracking-widest uppercase px-1.5 py-0.5 rounded-full" style={{ border: "1px solid #C9971A", color: "#C9971A" }}>
              ADMIN
            </span>
          )}
          <ChevronDown
            className={`h-4 w-4 text-gray-400 transition-transform duration-150 ${open ? "rotate-180" : ""}`}
          />
        </button>

        {/* Dropdown */}
        {open && (
          <div className="absolute right-0 mt-2 w-60 rounded-xl border bg-white shadow-xl py-1 z-50" style={{ borderColor: "#ECE6D8", boxShadow: "0 8px 30px rgba(60,45,10,0.10)" }}>
            {/* Identity header */}
            <div className="px-4 py-3 border-b" style={{ borderColor: "#ECE6D8" }}>
              <div className="flex items-center gap-3">
                <div className="h-9 w-9 rounded-full flex items-center justify-center text-white text-sm font-bold flex-shrink-0" style={{ background: "#151515" }}>
                  {initials}
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-gray-900 dark:text-gray-100 truncate">
                    {profile.username}
                  </p>
                  {user?.email && (
                    <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
                      {user.email}
                    </p>
                  )}
                </div>
              </div>
              <span
                className="mt-2.5 inline-block rounded-full px-2.5 py-0.5 text-[10px] font-bold tracking-widest uppercase"
                style={isAdmin
                  ? { border: "1px solid #C9971A", color: "#C9971A" }
                  : { background: "#F5F0E4", color: "#6B6660" }}
              >
                {isAdmin ? "ADMIN" : "Advocate"}
              </span>
            </div>

            {/* Sign out */}
            <button
              onClick={() => { setOpen(false); setConfirming(true); }}
              className="w-full flex items-center gap-2.5 px-4 py-2.5 text-sm text-red-600 hover:bg-red-50 transition-colors"
            >
              <LogOut className="h-4 w-4" />
              Sign out
            </button>
          </div>
        )}
      </div>

      {/* Confirmation modal — rendered outside the relative div so it covers full screen */}
      {confirming && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm"
          onClick={(e) => { if (e.target === e.currentTarget) setConfirming(false); }}
        >
          <div className="w-full max-w-sm mx-4 rounded-2xl border bg-white p-6 shadow-2xl" style={{ borderColor: "#ECE6D8" }}>
            <div className="mb-1 flex items-center gap-2.5">
              <div className="h-9 w-9 rounded-full bg-red-100 flex items-center justify-center flex-shrink-0">
                <LogOut className="h-4 w-4 text-red-600 dark:text-red-400" />
              </div>
              <h2 className="text-base font-semibold text-gray-900">Sign out?</h2>
            </div>
            <p className="mt-1 text-sm text-gray-500 pl-[52px]">
              You&apos;ll be returned to the sign-in page.
            </p>
            <div className="mt-5 flex gap-2.5 justify-end">
              <button
                onClick={() => setConfirming(false)}
                className="px-4 py-2 text-sm font-medium rounded-lg border text-gray-700 hover:bg-gray-50 transition-colors"
                style={{ borderColor: "#ECE6D8" }}
              >
                Cancel
              </button>
              <button
                onClick={handleSignOut}
                className="px-4 py-2 text-sm font-medium rounded-lg bg-red-600 hover:bg-red-700 text-white transition-colors"
              >
                Sign out
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
