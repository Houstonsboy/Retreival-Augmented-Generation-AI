import { SkeletonBar } from "./SkeletonBar";
import { SkeletonPill } from "./SkeletonPill";

/**
 * Mirrors SheetHeader:
 *   px-8 pt-7 pb-4 | kind badge + copy/link icons | h2 title (text-2xl) | subtitle italic | court/year | jump chips
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
          {/* Kind badge: text-[9px] uppercase px-2 py-0.5 rounded-md */}
          <SkeletonPill width="56px" height="1.25rem" />
          {/* Relevance score */}
          <SkeletonBar width="72px" height="0.75rem" />
        </div>
        {/* Action buttons: copy + external link */}
        <div className="flex items-center gap-2 shrink-0">
          <SkeletonBar width="1.75rem" height="1.75rem" radius="0.5rem" />
          <SkeletonBar width="1.75rem" height="1.75rem" radius="0.5rem" />
        </div>
      </div>

      {/* h2 title: text-2xl font-bold leading-tight (line-h ≈ 30px, bar ≈ 18px) */}
      <div className="flex flex-col gap-2 mb-1.5">
        <SkeletonBar width="70%" height="var(--sk-title-h)" radius="var(--sk-radius-lg)" />
        <SkeletonBar width="42%" height="var(--sk-title-h)" radius="var(--sk-radius-lg)" />
      </div>

      {/* Subtitle: text-sm italic */}
      <SkeletonBar width="30%" height="0.75rem" style={{ marginBottom: "0.5rem" }} />

      {/* Court / year: text-xs */}
      <SkeletonBar width="22%" height="0.65rem" style={{ marginBottom: "1rem" }} />

      {/* Jump-to chips row */}
      <div className="flex flex-wrap gap-1.5">
        <SkeletonBar width="100px" height="0.6rem" radius="2px" style={{ alignSelf: "center" }} />
        <SkeletonPill width="64px" height="1.5rem" />
        <SkeletonPill width="74px" height="1.5rem" />
        <SkeletonPill width="58px" height="1.5rem" />
      </div>
    </div>
  );
}
