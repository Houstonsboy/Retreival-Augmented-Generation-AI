import { SkeletonBar } from "./SkeletonBar";

// Widths cycle to produce the ragged-right-edge look
const WIDTHS = ["100%", "97%", "92%", "95%", "89%", "98%", "91%"];

interface SkeletonTextProps {
  lines?: number;
  gap?: string;
  lastLineWidth?: string;
  /** Height of each bar. Defaults to CSS var --sk-line-h */
  barHeight?: string;
  className?: string;
}

export function SkeletonText({
  lines = 4,
  gap = "0.55rem",
  lastLineWidth = "52%",
  barHeight = "var(--sk-line-h)",
  className = "",
}: SkeletonTextProps) {
  return (
    <div
      aria-hidden="true"
      className={className}
      style={{ display: "flex", flexDirection: "column", gap }}
    >
      {Array.from({ length: lines }, (_, i) => (
        <SkeletonBar
          key={i}
          width={i === lines - 1 ? lastLineWidth : WIDTHS[i % WIDTHS.length]}
          height={barHeight}
        />
      ))}
    </div>
  );
}
