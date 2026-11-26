export type UserType = "advocate" | "admin" | "litigant";

/**
 * Paths that anyone — including unauthenticated visitors — may open.
 *
 * Static assets (/_next/*, /favicon.ico, /robots.txt) are served directly by
 * Next.js before any React component runs, so AuthGuard never sees them; no
 * need to list them here.
 */
export const PUBLIC_PATHS = [
  "/",
  "/auth/sign-in",
  "/auth/sign-up",
  "/auth/complete-profile",
  "/litigant/sign-in",
] as const;

/** True when an unauthenticated user should be allowed to view `pathname`. */
export function isPublicPath(pathname: string): boolean {
  return PUBLIC_PATHS.some((p) =>
    p === "/" ? pathname === "/" : pathname.startsWith(p)
  );
}

// Paths only admins may open.
export const ADMIN_ONLY_PATHS = ["/ingester", "/firacChecker", "/constitution"];

// Paths only advocate/admin may open (litigants cannot enter).
const ADVOCATE_ONLY_ROUTES = ["/research", "/chat/", "/dashboard", "/repository"];

/**
 * Returns true if `type` is permitted on `pathname`.
 * Public paths and /litigant/sign-in are NOT checked here — handle them first
 * in AuthGuard before calling this function.
 *
 * Access map (default: DENY — unknown paths are not reachable):
 *   /litigant/*                              → litigant, advocate, admin
 *   /research, /chat/*, /dashboard, /repository/* → advocate, admin
 *   /ingester, /firacChecker, /constitution  → admin only
 *   anything else                            → denied (→ homeRoute)
 */
export function canAccess(pathname: string, type: UserType): boolean {
  if (ADMIN_ONLY_PATHS.some((p) => pathname.startsWith(p))) {
    return type === "admin";
  }
  if (pathname.startsWith("/litigant")) {
    return true; // all three types
  }
  if (ADVOCATE_ONLY_ROUTES.some((p) => pathname.startsWith(p))) {
    return type === "advocate" || type === "admin";
  }
  return false; // default DENY
}

/** The home path for each role after authentication. */
export function homeRoute(type: UserType): string {
  return type === "litigant" ? "/litigant/analysis" : "/dashboard";
}
