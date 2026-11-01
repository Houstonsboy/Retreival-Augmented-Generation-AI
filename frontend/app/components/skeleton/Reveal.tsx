import type { CSSProperties, ReactNode } from "react";

interface RevealProps {
  children: ReactNode;
  /** Stagger delay in ms */
  delay?: number;
  className?: string;
  style?: CSSProperties;
}

/**
 * Wraps content in a CSS-only reveal animation (opacity 0→1, translateY 4→0).
 * Uses animation-delay so staggered siblings don't need any JS.
 * Works in both server and client components.
 */
export function Reveal({ children, delay = 0, className = "", style }: RevealProps) {
  const combined: CSSProperties = {
    animationDelay: delay > 0 ? `${delay}ms` : undefined,
    ...style,
  };
  return (
    <div className={`sk-reveal ${className}`} style={combined}>
      {children}
    </div>
  );
}
