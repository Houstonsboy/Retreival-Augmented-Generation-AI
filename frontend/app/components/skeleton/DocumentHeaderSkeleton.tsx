import { Skeleton } from "@/app/components/loading/Skeleton";

/**
 * Mirrors SheetHeader: px-8 pt-7 pb-4 | kind badge + copy/link icons |
 * h2 title (text-2xl) | subtitle italic | court/year | jump chips
 */
export function DocumentHeaderSkeleton() {
  return (
    <div
      aria-hidden="true"
      className="px-8 pt-7 pb-4"
      style={{ borderBottom: "1px solid #ECE6D8" }}
    >
      {/* Top meta row: kind badge + action icons */}
      <div className="flex items-start justify-between gap-4 mb-3">
        <div className="flex items-center gap-2">
          <Skeleton variant="pill"  width="56px" />
          <Skeleton variant="line"  width="72px" height="0.75rem" />
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Skeleton variant="line" width="1.75rem" height="1.75rem" radius="0.5rem" />
          <Skeleton variant="line" width="1.75rem" height="1.75rem" radius="0.5rem" />
        </div>
      </div>

      {/* h2 title: text-2xl font-bold */}
      <div className="flex flex-col gap-2 mb-1.5">
        <Skeleton variant="title" width="70%" />
        <Skeleton variant="title" width="42%" />
      </div>

      {/* Subtitle: text-sm italic */}
      <Skeleton variant="line" width="30%" height="0.75rem" style={{ marginBottom: "0.5rem" }} />

      {/* Court / year: text-xs */}
      <Skeleton variant="line" width="22%" height="0.65rem" style={{ marginBottom: "1rem" }} />

      {/* Jump-to chips row */}
      <div className="flex flex-wrap gap-1.5">
        <Skeleton variant="line" width="100px" height="0.6rem" radius="2px" style={{ alignSelf: "center" }} />
        <Skeleton variant="pill" width="64px" />
        <Skeleton variant="pill" width="74px" />
        <Skeleton variant="pill" width="58px" />
      </div>
    </div>
  );
}
