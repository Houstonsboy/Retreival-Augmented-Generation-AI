import { use } from "react";
import { ChatView } from "@/app/components/ChatView";

export default function ConvPage({ params }: { params: Promise<{ conversationId: string }> }) {
  const { conversationId } = use(params);
  return <ChatView initialConversationId={conversationId} />;
}
