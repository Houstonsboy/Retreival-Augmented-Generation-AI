import { SkeletonBar } from "./SkeletonBar";
import { SkeletonCircle } from "./SkeletonCircle";
import { SkeletonPill } from "./SkeletonPill";

/**
 * Mirrors the <header> in ChatView exactly:
 *   px-5 py-2.5 | menu button (w-8 h-8) | logo (w-8 h-8 rounded-xl) | title bar | dashboard pill
 *   Right: avatar circle + username pill
 */
export function TopBarSkeleton() {
  return (
    <header
      aria-hidden="true"
      className="shrink-0 z-20"
      style={{ backgroundColor: "#FAF7F0", borderBottom: "1px solid #ECE6D8" }}
    >
      <div className="px-5 py-2.5 flex items-center justify-between">
        {/* Left cluster */}
        <div className="flex items-center gap-3">
          {/* Hamburger button: w-8 h-8 rounded-lg */}
          <SkeletonBar width="2rem" height="2rem" radius="0.5rem" />
          {/* Logo: w-8 h-8 rounded-xl */}
          <SkeletonBar width="2rem" height="2rem" radius="0.75rem" />
          {/* App title: text-sm font-bold */}
          <SkeletonBar width="110px" height="0.875rem" />
          {/* Dashboard link pill */}
          <SkeletonPill width="90px" height="1.5rem" />
        </div>
        {/* Right cluster: avatar + dropdown chevron area */}
        <div className="flex items-center gap-2">
          <SkeletonCircle size="2rem" />
          <SkeletonBar width="72px" height="0.75rem" />
        </div>
      </div>
    </header>
  );
}
