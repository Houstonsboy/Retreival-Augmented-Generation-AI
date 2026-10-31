"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

// Worker setup — must be client-side only
if (typeof window !== "undefined") {
  pdfjs.GlobalWorkerOptions.workerSrc = new URL(
    "pdfjs-dist/build/pdf.worker.min.mjs",
    import.meta.url
  ).toString();
}

export interface BboxEntry {
  page: number;   // 0-indexed PDF page
  bbox: [number, number, number, number]; // x0,y0,x1,y1 in PDF points
}

export interface ChunkLocation {
  chunk_id: string;
  page_start: number; // 1-indexed
  page_end: number;
  boxes: BboxEntry[];
  page_dims?: { width: number; height: number }[];
  ocr: boolean;
}

interface Props {
  pdfUrl: string;
  locations: ChunkLocation[];           // all chunks for this doc
  citedChunkIds: Set<string>;           // solid gold highlight
  unusedChunkIds?: Set<string>;         // dashed outline only
  onPageChange?: (page: number) => void;
  initialPage?: number;
  scrollToChunkId?: string | null;
}

const GOLD         = "#C9971A";
const HIGHLIGHT_BG = "rgba(246,231,184,0.55)";
const UNUSED_BG    = "rgba(200,200,200,0.15)";

export default function PdfViewer({
  pdfUrl,
  locations,
  citedChunkIds,
  unusedChunkIds = new Set(),
  onPageChange,
  initialPage = 1,
  scrollToChunkId,
}: Props) {
  const [numPages, setNumPages] = useState<number>(0);
  const [containerWidth, setContainerWidth] = useState<number>(0);
  const [pageDims, setPageDims] = useState<Record<number, { width: number; height: number }>>({});
  const [renderWidths, setRenderWidths] = useState<Record<number, number>>({});
  const containerRef = useRef<HTMLDivElement>(null);
  const pageRefs = useRef<Record<number, HTMLDivElement | null>>({});

  // Measure container
  useEffect(() => {
    const measure = () => {
      if (containerRef.current) {
        setContainerWidth(containerRef.current.clientWidth);
      }
    };
    measure();
    const ro = new ResizeObserver(measure);
    if (containerRef.current) ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  const onDocumentLoadSuccess = ({ numPages }: { numPages: number }) => {
    setNumPages(numPages);
  };

  const onPageLoadSuccess = useCallback((pageNum: number, page: any) => {
    const vp = page.getViewport({ scale: 1 });
    setPageDims((prev) => ({ ...prev, [pageNum]: { width: vp.width, height: vp.height } }));
  }, []);

  const onPageRenderSuccess = useCallback((pageNum: number) => {
    const el = pageRefs.current[pageNum];
    if (el) {
      setRenderWidths((prev) => ({ ...prev, [pageNum]: el.clientWidth }));
    }
  }, []);

  // Scroll to chunk
  useEffect(() => {
    if (!scrollToChunkId) return;
    const loc = locations.find((l) => l.chunk_id === scrollToChunkId);
    if (!loc) return;
    const targetPage = loc.page_start;
    const el = pageRefs.current[targetPage];
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [scrollToChunkId, locations]);

  // Build bbox overlay map: pageNum → list of overlay rects
  const overlaysByPage: Record<number, Array<{ rect: [number, number, number, number]; isCited: boolean; chunkId: string }>> = {};
  for (const loc of locations) {
    const isCited = citedChunkIds.has(loc.chunk_id);
    const isUnused = unusedChunkIds.has(loc.chunk_id);
    if (!isCited && !isUnused) continue;
    for (const entry of loc.boxes) {
      const pg = entry.page + 1; // convert to 1-indexed
      if (!overlaysByPage[pg]) overlaysByPage[pg] = [];
      overlaysByPage[pg].push({ rect: entry.bbox, isCited, chunkId: loc.chunk_id });
    }
    // Page-level fallback (no boxes)
    if (!loc.boxes.length && loc.page_start) {
      const pg = loc.page_start;
      if (!overlaysByPage[pg]) overlaysByPage[pg] = [];
      overlaysByPage[pg].push({ rect: [0, 0, 0, 0] as [number, number, number, number], isCited, chunkId: loc.chunk_id });
    }
  }

  const getScale = (pageNum: number): number => {
    const renderW = renderWidths[pageNum] ?? containerWidth;
    const pdfW = pageDims[pageNum]?.width ?? 612;
    return renderW / pdfW;
  };

  const renderOverlays = (pageNum: number) => {
    const items = overlaysByPage[pageNum];
    if (!items?.length) return null;
    const scale = getScale(pageNum);
    const pdfH = pageDims[pageNum]?.height ?? 792;
    return (
      <>
        {items.map((item, i) => {
          const [x0, y0, x1, y1] = item.rect;
          if (x0 === 0 && y0 === 0 && x1 === 0 && y1 === 0) {
            // Page-level match chip
            return (
              <div key={i}
                className="absolute top-2 right-2 px-2 py-0.5 rounded text-[10px] font-semibold"
                style={{ background: HIGHLIGHT_BG, border: `1px solid ${GOLD}`, color: GOLD }}>
                page-level match
              </div>
            );
          }
          const left   = x0 * scale;
          const top    = y0 * scale;
          const width  = (x1 - x0) * scale;
          const height = (y1 - y0) * scale;
          return (
            <div key={i} className="absolute pointer-events-none"
              style={{
                left, top, width, height,
                background: item.isCited ? HIGHLIGHT_BG : UNUSED_BG,
                borderLeft: item.isCited ? `2px solid ${GOLD}` : undefined,
                border: !item.isCited ? `1px dashed rgba(180,180,180,0.6)` : undefined,
              }}
            />
          );
        })}
      </>
    );
  };

  return (
    <div ref={containerRef} className="relative w-full overflow-y-auto" style={{ background: "#F5F5F5" }}>
      {containerWidth > 0 && (
        <Document
          file={pdfUrl}
          onLoadSuccess={onDocumentLoadSuccess}
          loading={
            <div className="flex items-center justify-center h-48 text-sm text-gray-400">
              Loading PDF…
            </div>
          }
          error={
            <div className="flex flex-col items-center justify-center h-48 gap-3 text-sm text-gray-400">
              <p>Could not load PDF.</p>
              <a href={pdfUrl} target="_blank" rel="noopener noreferrer"
                className="underline text-amber-700">Open in new tab</a>
            </div>
          }
        >
          {Array.from({ length: numPages }, (_, i) => i + 1).map((pageNum) => (
            <div
              key={pageNum}
              ref={(el) => { pageRefs.current[pageNum] = el; }}
              className="relative mb-2 mx-auto shadow-sm"
              style={{ width: "100%", background: "white" }}
            >
              <Page
                pageNumber={pageNum}
                width={containerWidth}
                onLoadSuccess={(page) => onPageLoadSuccess(pageNum, page)}
                onRenderSuccess={() => onPageRenderSuccess(pageNum)}
                renderAnnotationLayer={false}
                renderTextLayer={false}
              />
              <div className="absolute inset-0 pointer-events-none">
                {renderOverlays(pageNum)}
              </div>
            </div>
          ))}
        </Document>
      )}
    </div>
  );
}
