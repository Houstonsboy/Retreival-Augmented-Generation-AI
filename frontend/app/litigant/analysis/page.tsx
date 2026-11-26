import { SCCVisionTest } from "@/app/components/sccvision/SCCVisionTest";

export default async function EvidenceAnalysisPage({
  searchParams,
}: {
  searchParams: Promise<{ caseId?: string }>;
}) {
  const { caseId } = await searchParams;
  return <SCCVisionTest key={caseId ?? "new"} initialCaseId={caseId} />;
}
