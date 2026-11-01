import { TopBarSkeleton } from "./TopBarSkeleton";
import { ChatPanelSkeleton } from "./ChatPanelSkeleton";
import { DocumentHeaderSkeleton } from "./DocumentHeaderSkeleton";
import { DocumentBodySkeleton } from "./DocumentBodySkeleton";
import { SourcesCardSkeleton } from "./SourcesCardSkeleton";
import { SkeletonBar } from "./SkeletonBar";
import { SkeletonPill } from "./SkeletonPill";

/**
 * Full-page skeleton for the chat workspace.
 * Used in loading.tsx files and as the ChatView internal loading state.
 *
 * showWorkspace=false → chat panel fills full width (home / new-chat route)
 * showWorkspace=true  → 40/60 split with document viewer (conversation route)
 */
export function WorkspaceSkeleton({ showWorkspace = true }: { showWorkspace?: boolean }) {
  return (
    <div
      aria-busy="true"
      role="main"
      className="flex h-screen flex-col"
      style={{ backgroundColor: "#FAF7F0", fontFamily: "var(--font-manrope), system-ui, sans-serif" }}
    >
      {/* Screen-reader announce (live region) */}
      <p className="sr-only" role="status" aria-live="polite">Loading document…</p>

      <TopBarSkeleton />

      <div className="flex-1 flex overflow-hidden">

        {/* ── LEFT: chat column ─────────────────────────────────────── */}
        <div
          className="flex flex-col overflow-hidden"
          style={{
            width: showWorkspace ? "40%" : "100%",
            minWidth: showWorkspace ? 260 : undefined,
            flexShrink: 0,
          }}
        >
          <ChatPanelSkeleton />
        </div>

        {/* ── Resizer placeholder ───────────────────────────────────── */}
        {showWorkspace && (
          <div style={{ width: 8, flexShrink: 0, background: "transparent" }} />
        )}

        {/* ── RIGHT: document workspace ─────────────────────────────── */}
        {showWorkspace && (
          <div
            className="flex-1 flex flex-col overflow-hidden"
            style={{ backgroundColor: "#F5F0E4" }}
          >
            {/* Workspace column header */}
            <div
              className="shrink-0 flex items-start justify-between px-5 py-3"
              style={{ borderBottom: "1px solid #ECE6D8", background: "#F5F0E4" }}
              aria-hidden="true"
            >
              <div style={{ display: "flex", flexDirection: "column", gap: "0.35rem" }}>
                {/* "RETRIEVED AUTHORITIES" eyebrow */}
                <SkeletonBar width="130px" height="0.6rem" />
                {/* Matter title */}
                <SkeletonBar width="200px" height="0.75rem" />
              </div>
              {/* ClusterWidget / Sources card */}
              <SourcesCardSkeleton />
            </div>

            {/* Tab bar: Reader / PDF */}
            <div
              className="shrink-0 flex items-center gap-2 px-5 py-2"
              style={{ borderBottom: "1px solid #ECE6D8", background: "#F5F0E4" }}
              aria-hidden="true"
            >
              <div
                className="flex rounded-full overflow-hidden"
                style={{ border: "1px solid #ECE6D8" }}
              >
                <SkeletonBar width="60px" height="1.75rem" radius="0" />
                <SkeletonBar width="42px" height="1.75rem" radius="0" style={{ marginLeft: 1 }} />
              </div>
              <div style={{ marginLeft: "auto" }}>
                <SkeletonPill width="80px" height="1.5rem" />
              </div>
            </div>

            {/* Document card */}
            <div className="flex-1 overflow-y-auto">
              <div className="px-6 py-8 flex justify-center">
                <div
                  className="w-full rounded-2xl bg-white"
                  style={{
                    maxWidth: "42rem",
                    border: "1px solid #ECE6D8",
                    boxShadow: "0 8px 32px rgba(60,45,10,0.08)",
                  }}
                >
                  <DocumentHeaderSkeleton />
                  <DocumentBodySkeleton />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
