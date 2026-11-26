export const MAX_FILES_PER_BATCH = 6;
export const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB — matches backend SCCVision/router.py

export const ACCEPTED_EXTENSIONS = [
  ".png", ".jpg", ".jpeg", ".webp",
] as const;

export const ACCEPTED_MIME_TYPES = new Set([
  "image/png",
  "image/jpeg",
  "image/webp",
]);

export function isAcceptedExtension(filename: string): boolean {
  const lower = filename.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

