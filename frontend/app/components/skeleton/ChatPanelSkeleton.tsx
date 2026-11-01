import { Skeleton, SkeletonParagraph } from "@/app/components/loading/Skeleton";

/** Right-aligned user message bubble skeleton */
function UserBubbleSkeleton({ lines = 2, lastW = "40%" }: { lines?: number; lastW?: string }) {
  return (
    <div aria-hidden="true" className="flex flex-col items-end">
      <div
        className="max-w-[88%] rounded-2xl px-4 py-3"
        style={{ background: "#F3E9CF" }}
      >
        <SkeletonParagraph
          lines={lines}
          lastWidth={lastW}
          gap="0.45rem"
          tone="accent"
        />
      </div>
    </div>
  );
}

/** Assistant response card skeleton */
function AssistantCardSkeleton({ bodyLines = 4 }: { bodyLines?: number }) {
  return (
    <div
      aria-hidden="true"
      className="rounded-2xl overflow-hidden"
      style={{ background: "#FFFDF8", border: "1px solid #ECE6D8" }}
    >
      {/* Card header */}
      <div
        className="flex items-center gap-2 px-4 py-3"
        style={{ borderBottom: "1px solid #ECE6D8" }}
      >
        <Skeleton variant="line" width="1.5rem" height="1.5rem" radius="0.5rem" />
        <Skeleton variant="line" width="110px" height="0.875rem" />
        <Skeleton variant="pill" width="145px" />
      </div>

      {/* Card body */}
      <div className="px-4 py-4">
        <SkeletonParagraph lines={bodyLines} lastWidth="48%" />
        {/* Bottom citations row */}
        <div
          className="flex items-center gap-2 mt-4 pt-3"
          style={{ borderTop: "1px solid #ECE6D8" }}
        >
          <Skeleton variant="line" width="50px"  height="0.6rem" />
          <Skeleton variant="pill" width="68px"  height="1.35rem" />
          <Skeleton variant="pill" width="80px"  height="1.35rem" />
          <Skeleton variant="pill" width="58px"  height="1.35rem" />
        </div>
      </div>

      {/* "Open sources" footer */}
      <div
        className="px-4 py-2.5 flex items-center gap-2"
        style={{ borderTop: "1px solid #ECE6D8" }}
      >
        <Skeleton variant="line" width="1rem"  height="1rem"   radius="3px" />
        <Skeleton variant="line" width="120px" height="0.75rem" />
      </div>
    </div>
  );
}

/**
 * Full left-panel skeleton. Mirrors py-5 space-y-6 px-4 message list
 * + fixed input bar at bottom.
 */
export function ChatPanelSkeleton() {
  return (
    <div className="flex flex-col h-full overflow-hidden" aria-busy="true">
      {/* Message stream */}
      <div className="flex-1 overflow-y-auto">
        <div className="py-5 space-y-6 px-4">
          <UserBubbleSkeleton lines={2} lastW="38%" />
          <AssistantCardSkeleton bodyLines={4} />
          <UserBubbleSkeleton lines={1} lastW="100%" />
          <AssistantCardSkeleton bodyLines={5} />
        </div>
      </div>

      {/* Composer placeholder */}
      <div
        className="shrink-0 px-4 py-3"
        style={{ borderTop: "1px solid #ECE6D8", background: "white" }}
      >
        <div className="flex items-end gap-3">
          <div
            className="flex-1 flex items-start rounded-2xl px-4 py-3"
            style={{ border: "1px solid #ECE6D8", background: "white" }}
          >
            <Skeleton variant="line" width="160px" height="0.875rem" style={{ marginTop: "2px" }} />
          </div>
          {/* Send button — dimmed */}
          <div
            className="w-10 h-10 rounded-full flex items-center justify-center"
            aria-hidden="true"
            style={{ background: "#C4B49A", opacity: 0.5 }}
          >
            <svg className="h-4 w-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 10l7-7m0 0l7 7m-7-7v18" />
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}
