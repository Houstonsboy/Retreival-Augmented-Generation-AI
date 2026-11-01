import { Skeleton, SkeletonParagraph } from "@/app/components/loading/Skeleton";

interface ParagraphSkeletonProps {
  lines:         number;
  lastLineWidth: string;
  highlighted?:  boolean;
  numbered?:     boolean;
}

function ParagraphSkeleton({ lines, lastLineWidth, highlighted = false, numbered = false }: ParagraphSkeletonProps) {
  const inner = (
    <div style={{ display: "flex", gap: "1rem" }}>
      {/* Gutter: w-6 text-right — small square for numbered subsections */}
      <div style={{ width: "1.5rem", flexShrink: 0, paddingTop: "0.15rem" }} aria-hidden="true">
        {numbered && <Skeleton variant="line" width="1rem" height="0.875rem" radius="2px" />}
      </div>
      {/* Paragraph text: text-[15px] leading-[1.75] */}
      <div style={{ flex: 1 }}>
        <SkeletonParagraph
          lines={lines}
          lastWidth={lastLineWidth}
          gap="0.65rem"
          tone={highlighted ? "accent" : "neutral"}
        />
      </div>
    </div>
  );

  if (highlighted) {
    return (
      <div
        aria-hidden="true"
        style={{
          background: "var(--skeleton-accent-base)",
          borderLeft: "3px solid #E2C673",
          borderRadius: "0.625rem",
          padding: "0.75rem 1rem",
          marginBottom: "0.25rem",
        }}
      >
        {inner}
      </div>
    );
  }

  return (
    <div aria-hidden="true" style={{ padding: "0.25rem 1rem", marginBottom: "0.75rem" }}>
      {inner}
    </div>
  );
}

/**
 * Mirrors the DocumentReader chunk list:
 * px-8 py-6 with paragraphs of text-[15px] leading-[1.75].
 */
export function DocumentBodySkeleton() {
  const paragraphs: Array<{ lines: number; last: string; highlighted: boolean; numbered: boolean }> = [
    { lines: 4, last: "55%", highlighted: false, numbered: false },
    { lines: 3, last: "45%", highlighted: false, numbered: true  },
    { lines: 5, last: "60%", highlighted: true,  numbered: true  },
    { lines: 3, last: "50%", highlighted: false, numbered: false },
    { lines: 4, last: "40%", highlighted: true,  numbered: true  },
    { lines: 3, last: "65%", highlighted: false, numbered: false },
  ];

  return (
    <div aria-hidden="true" className="px-8 py-6">
      {/* "Full document" info banner */}
      <div
        className="flex items-center gap-2 mb-4 px-0 py-1.5"
        style={{ borderBottom: "1px solid #ECE6D8" }}
      >
        <Skeleton variant="line" width="90px"  height="0.65rem" />
        <Skeleton variant="line" width="160px" height="0.65rem" />
      </div>

      {paragraphs.map((p, i) => (
        <ParagraphSkeleton
          key={i}
          lines={p.lines}
          lastLineWidth={p.last}
          highlighted={p.highlighted}
          numbered={p.numbered}
        />
      ))}
    </div>
  );
}
