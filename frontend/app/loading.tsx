/**
 * Route-level loading for "/" (new-chat / home).
 * Shown by Next.js App Router while the page JS chunk is loading.
 * No workspace panel — home starts with the empty-state input.
 */
import { WorkspaceSkeleton } from "@/app/components/skeleton/WorkspaceSkeleton";

export default function HomeLoading() {
  return <WorkspaceSkeleton showWorkspace={false} />;
}
