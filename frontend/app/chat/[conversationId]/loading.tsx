/**
 * Route-level loading for "/chat/[conversationId]".
 * Shown while the page JS chunk is loading (deep link, browser refresh,
 * or first navigation to a conversation URL).
 * Renders the full two-column workspace skeleton.
 */
import { WorkspaceSkeleton } from "@/app/components/skeleton/WorkspaceSkeleton";

export default function ConvLoading() {
  return <WorkspaceSkeleton showWorkspace={true} />;
}
