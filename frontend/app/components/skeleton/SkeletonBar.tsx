import type { CSSProperties } from "react";

interface SkeletonBarProps {
  width?: string | number;
  height?: string | number;
  radius?: string;
  className?: string;
  style?: CSSProperties;
}

export function SkeletonBar({
  width = "100%",
  height = "var(--sk-line-h)",
  radius = "var(--sk-radius)",
  className = "",
  style,
}: SkeletonBarProps) {
  return (
    <span
      aria-hidden="true"
      className={`sk-bar ${className}`}
      style={{ width, height, borderRadius: radius, ...style }}
    />
  );
}
