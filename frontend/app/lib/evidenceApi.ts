import { apiFetch, fetchProtectedFile } from "./api";
import type { CertificateDetail, CertificateRecord, DeviceFacts } from "./evidenceTypes";

export async function listEvidence(): Promise<never[]> {
  // TODO(backend): GET /api/evidence — returns EvidenceRecord[]
  return [];
}

export async function listCertificates(): Promise<CertificateRecord[]> {
  const res = await apiFetch("/api/evidence/certificates");
  if (!res.ok) return [];
  const body = await res.json().catch(() => ({}));
  return (body.certificates as CertificateRecord[]) ?? [];
}

export async function getCertificate(certId: string): Promise<CertificateDetail | null> {
  const res = await apiFetch(`/api/evidence/certificates/${certId}`);
  if (!res.ok) return null;
  return res.json();
}

export async function createCertificate(
  analysisId: string,
  deviceFacts: DeviceFacts,
): Promise<CertificateRecord> {
  const res = await apiFetch("/api/evidence/certificates", {
    method: "POST",
    body: JSON.stringify({ analysis_id: analysisId, device_facts: deviceFacts }),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const details = body.details as Array<{ loc: string[]; msg: string }> | undefined;
    const detail = details?.[0]?.msg ?? body.error ?? `Request failed (${res.status})`;
    throw new Error(detail);
  }
  return body as CertificateRecord;
}

export async function downloadCertificate(certId: string, certNumber: string): Promise<void> {
  const blobUrl = await fetchProtectedFile(`/api/evidence/certificates/${certId}/download`);
  const a = document.createElement("a");
  a.href = blobUrl;
  a.download = `${certNumber}.pdf`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

export async function viewCertificate(certId: string): Promise<void> {
  const blobUrl = await fetchProtectedFile(`/api/evidence/certificates/${certId}/download`);
  window.open(blobUrl, "_blank", "noopener,noreferrer");
}
