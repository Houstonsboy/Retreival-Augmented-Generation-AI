import type { CSSProperties, ReactNode } from "react";

export type SkeletonVariant = "line" | "title" | "pill" | "block" | "circle";
export type SkeletonTone    = "neutral" | "accent";

interface SkeletonProps {
  variant?:   SkeletonVariant;
  tone?:      SkeletonTone;
  /** Explicit width override; e.g. "60%", "120px", "100%" */
  width?:     string | number;
  /** Explicit height override */
  height?:    string | number;
  /** Border radius override */
  radius?:    string | number;
  /** Used by variant="circle" to set equal width and height */
  size?:      number;
  /** Extra CSS class names */
  className?: string;
  style?:     CSSProperties;
  /** Only valid when variant="block" — children render inside the block */
  children?:  ReactNode;
}

const DEFAULTS: Record<SkeletonVariant, { h: string; r: string }> = {
  line:   { h: "11px",  r: "6px"    },
  title:  { h: "19px",  r: "6px"    },
  pill:   { h: "20px",  r: "9999px" },
  block:  { h: "auto",  r: "8px"    },
  circle: { h: "auto",  r: "50%"    },
};

/** Widths to cycle through for natural paragraph look */
export const LINE_WIDTHS = ["100%", "92%", "85%", "96%", "88%", "78%", "94%"];

/**
 * Unified skeleton primitive.
 *
 * All instances share one viewport-anchored shimmer wave via
 * background-attachment: fixed — they animate in perfect sync.
 *
 * variant="block" renders a <div> so children can sit inside it; all
 * other variants render a self-closing <span>.
 */
export function Skeleton({
  variant   = "line",
  tone      = "neutral",
  width,
  height,
  radius,
  size,
  className = "",
  style,
  children,
}: SkeletonProps) {
  const def     = DEFAULTS[variant];
  const isCircle = variant === "circle";
  const isBlock  = variant === "block";

  const w  = isCircle ? `${size ?? 24}px`        : (width  ?? undefined);
  const h  = isCircle ? `${size ?? 24}px`         : (height ?? (isBlock ? undefined : def.h));
  const r  = radius ?? def.r;
  const cls = [
    "skeleton",
    tone === "accent" ? "skeleton--accent" : "",
    className,
  ].filter(Boolean).join(" ");

  const css: CSSProperties = { borderRadius: r, ...style };
  if (w)  css.width     = w;
  if (h)  css.height    = h;
  if (!h && isBlock) css.minHeight = "2.5rem";

  if (isBlock) {
    return (
      <div aria-hidden="true" className={cls} style={css}>
        {children}
      </div>
    );
  }

  return (
    <span aria-hidden="true" className={cls} style={css} />
  );
}

/**
 * Convenience: a paragraph of N skeleton lines with naturally varying widths.
 * The last line is always shorter (lastWidth) to look like a real paragraph.
 */
export function SkeletonParagraph({
  lines       = 4,
  lastWidth   = "55%",
  gap         = "0.55rem",
  tone        = "neutral",
}: {
  lines?:     number;
  lastWidth?: string;
  gap?:       string;
  tone?:      SkeletonTone;
}) {
  return (
    <div aria-hidden="true" style={{ display: "flex", flexDirection: "column", gap }}>
      {Array.from({ length: lines }, (_, i) => (
        <Skeleton
          key={i}
          variant="line"
          tone={tone}
          width={i === lines - 1 ? lastWidth : LINE_WIDTHS[i % LINE_WIDTHS.length]}
        />
      ))}
    </div>
  );
}
