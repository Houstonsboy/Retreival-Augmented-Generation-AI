import { Skeleton } from "@/app/components/loading/Skeleton";

/** Mirrors the <header> px-5 py-2.5 layout in ChatView exactly. */
export function TopBarSkeleton() {
  return (
    <header
      aria-hidden="true"
      className="shrink-0 z-20"
      style={{ backgroundColor: "#FAF7F0", borderBottom: "1px solid #ECE6D8" }}
    >
      <div className="px-5 py-2.5 flex items-center justify-between">
        {/* Left: hamburger w-8 | logo w-8 rounded-xl | title | dashboard pill */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <Skeleton variant="line"  width="2rem"  height="2rem"   radius="0.5rem" />
          <Skeleton variant="line"  width="2rem"  height="2rem"   radius="0.75rem" />
          <Skeleton variant="title" width="110px" />
          <Skeleton variant="pill"  width="90px" />
        </div>
        {/* Right: avatar circle + username bar */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <Skeleton variant="circle" size={32} />
          <Skeleton variant="line"   width="72px" />
        </div>
      </div>
    </header>
  );
}
