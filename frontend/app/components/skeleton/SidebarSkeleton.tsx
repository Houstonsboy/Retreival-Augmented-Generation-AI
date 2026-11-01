import { Skeleton } from "@/app/components/loading/Skeleton";

const ITEM_WIDTHS = ["85%", "70%", "90%", "60%", "78%"];

/** Shown inside the sidebar drawer while conversations are loading. */
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
          <Skeleton variant="line" width={ITEM_WIDTHS[i % ITEM_WIDTHS.length]} height="0.75rem" />
          {/* Meta row: timestamp + turn count */}
          <div className="flex items-center justify-between mt-1.5">
            <Skeleton variant="line" width="55px" height="0.6rem" />
            <Skeleton variant="line" width="32px" height="0.6rem" />
          </div>
        </div>
      ))}
    </div>
  );
}
