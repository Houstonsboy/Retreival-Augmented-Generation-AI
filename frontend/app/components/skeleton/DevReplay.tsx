"use client";

/**
 * Dev-only "Replay loading" button.
 * Reloads the page (preserving current URL so ?simulateLatency=1 survives).
 * Renders NOTHING in production builds — tree-shaken out entirely.
 *
 * Usage:
 *   Add ?simulateLatency=1 to the URL, then use this button to re-trigger.
 *   Remove the param to stop the delays.
 *
 * The button is fixed bottom-right so it never interferes with layout.
 */
export function DevReplay() {
  if (process.env.NODE_ENV !== "development") return null;

  const handleReplay = () => {
    // Ensure ?simulateLatency=1 is in the URL for the reload
    const url = new URL(window.location.href);
    url.searchParams.set("simulateLatency", "1");
    window.location.href = url.toString();
  };

  const handleClear = () => {
    const url = new URL(window.location.href);
    url.searchParams.delete("simulateLatency");
    window.location.href = url.toString();
  };

  // Only show the button if we're in dev mode (checked at runtime too)
  return (
    <div
      style={{
        position: "fixed",
        bottom: 16,
        right: 16,
        zIndex: 9999,
        display: "flex",
        flexDirection: "column",
        gap: 4,
        alignItems: "flex-end",
      }}
    >
      <button
        onClick={handleReplay}
        title="Reload with ?simulateLatency=1 to replay skeleton loading"
        style={{
          background: "#1a1a2e",
          color: "#C9971A",
          border: "1px solid #C9971A44",
          borderRadius: 8,
          padding: "5px 12px",
          fontSize: 11,
          fontWeight: 700,
          cursor: "pointer",
          letterSpacing: "0.05em",
          opacity: 0.9,
        }}
      >
        ↺ Replay loading
      </button>
      <button
        onClick={handleClear}
        title="Remove ?simulateLatency=1 and reload normally"
        style={{
          background: "transparent",
          color: "#6B6660",
          border: "none",
          padding: "2px 6px",
          fontSize: 10,
          cursor: "pointer",
        }}
      >
        clear latency
      </button>
    </div>
  );
}
