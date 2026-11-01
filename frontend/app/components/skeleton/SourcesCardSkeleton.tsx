import { SkeletonBar } from "./SkeletonBar";
import { SkeletonPill } from "./SkeletonPill";

/**
 * Mirrors the ClusterWidget trigger button:
 *   rounded-full px-3 py-1.5 with stacked mini-icon squares + label + count badge
 */
export function SourcesCardSkeleton() {
  return (
    <div
      aria-hidden="true"
      className="flex items-center gap-2 rounded-full px-3 py-1.5"
      style={{ background: "white", border: "1px solid #ECE6D8" }}
    >
      {/* Stacked mini document icons (4 overlapping squares) */}
      <div style={{ position: "relative", width: "1.75rem", height: "1.25rem", flexShrink: 0 }}>
        {[0, 1, 2, 3].map((i) => (
          <SkeletonBar
            key={i}
            width="1.25rem"
            height="1.25rem"
            radius="3px"
            style={{
              position: "absolute",
              left: `${i * 4}px`,
              top: 0,
              zIndex: i,
            }}
          />
        ))}
      </div>

      {/* Label text */}
      <div style={{ display: "flex", flexDirection: "column", gap: "3px" }}>
        <SkeletonBar width="60px" height="0.7rem" />
        <SkeletonBar width="80px" height="0.55rem" />
      </div>

      {/* Count badge pill */}
      <SkeletonPill width="1rem" height="1rem" />
    </div>
  );
}
