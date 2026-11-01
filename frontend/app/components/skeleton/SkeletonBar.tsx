import type { CSSProperties } from "react";

interface SkeletonBarProps {
  width?:     string | number;
  height?:    string | number;
  radius?:    string;
  accent?:    boolean;
  className?: string;
  style?:     CSSProperties;
}

/** Single skeleton bar. Uses the shared .skeleton class for synchronized shimmer. */
export function SkeletonBar({
  width   = "100%",
  height  = "11px",
  radius  = "6px",
  accent  = false,
  className = "",
  style,
}: SkeletonBarProps) {
  return (
    <span
      aria-hidden="true"
      className={`skeleton${accent ? " skeleton--accent" : ""} ${className}`}
      style={{ display: "block", width, height, borderRadius: radius, flexShrink: 0, ...style }}
    />
  );
}
