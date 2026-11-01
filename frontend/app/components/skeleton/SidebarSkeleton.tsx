import { SkeletonBar } from "./SkeletonBar";
import { SkeletonPill } from "./SkeletonPill";

const ITEM_WIDTHS = ["85%", "70%", "90%", "60%", "78%"];

/**
 * Shown inside the sidebar drawer while conversations are loading.
 * Matches the conversation list item structure: title bar + meta row.
 */
export function SidebarSkeleton({ count = 5 }: { count?: number }) {
  return (
    <div aria-hidden="true" className="py-2 px-2 space-y-1">
      {Array.from({ length: count }, (_, i) => (
        <div
          key={i}
          className="rounded-xl px-3 py-2.5"
          style={{ border: "1.5px solid transparent" }}
        >
          {/* Title row */}
          <SkeletonBar width={ITEM_WIDTHS[i % ITEM_WIDTHS.length]} height="0.75rem" />
          {/* Meta row: timestamp + turn count */}
          <div className="flex items-center justify-between mt-1.5">
            <SkeletonBar width="55px" height="0.6rem" />
            <SkeletonBar width="32px" height="0.6rem" />
          </div>
        </div>
      ))}
    </div>
  );
}
