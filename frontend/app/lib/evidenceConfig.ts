export const MAX_FILES_PER_BATCH = 6;
export const MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024; // 15 MB

export const ACCEPTED_EXTENSIONS = [
  ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".heic", ".eml",
] as const;

export const ACCEPTED_MIME_TYPES = new Set([
  "application/pdf",
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/heic",
  "image/heif",
  "message/rfc822",    // .eml
  "application/octet-stream", // fallback for .heic / .eml on some OS
]);

export function isAcceptedExtension(filename: string): boolean {
  const lower = filename.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

