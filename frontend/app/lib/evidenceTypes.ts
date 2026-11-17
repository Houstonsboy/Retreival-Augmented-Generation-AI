export type EvidenceType =
  | "mpesa_statement"
  | "mpesa_sms"
  | "whatsapp_sms_chat"
  | "email"
  | "written_agreement_photo"
  | "other";

export const EVIDENCE_TYPE_LABELS: Record<EvidenceType, string> = {
  mpesa_statement: "M-Pesa statement",
  mpesa_sms: "M-Pesa SMS screenshot",
  whatsapp_sms_chat: "WhatsApp or SMS chat",
  email: "Email",
  written_agreement_photo: "Photo or scan of written agreement",
  other: "Other",
};

export type EvidenceStatus =
  | "uploaded"
  | "extracting"
  | "analysing"
  | "awaiting_answers"
  | "complete"
  | "failed";

export type CertificateOutcome =
  | "certificate_ready"
  | "needs_strengthening"
  | "weak";

// A file the user has selected but not yet submitted.
export interface EvidenceFileDraft {
  id: string;
  file: File;
  evidenceType: EvidenceType | "";
  passwordProtected: boolean;
  password: string;
  thumbnail: string | null; // data-URL for image previews, null otherwise
  processingStatus: EvidenceStatus | "queued" | "reading";
}

// A persisted evidence record returned from the backend.
export interface EvidenceRecord {
  id: string;
  fileName: string;
  evidenceType: EvidenceType;
  sizeBytes: number;
  uploadedAt: string; // ISO-8601
  status: EvidenceStatus;
}

// ── Section 106B device facts ─────────────────────────────────────────────

export type ProductionMethod =
  | "screenshot_on_device"
  | "forwarded_or_copied"
  | "printed_statement"
  | "other";

export interface DeviceFacts {
  device_make: string;
  device_model: string;
  phone_number_last4: string | null;
  production_method: ProductionMethod;
  owner_is_declarant: boolean;
  lawful_control: boolean;
  regular_use: boolean;
  regular_feed: boolean;
  operating_properly: boolean;
  malfunction_details: string | null;
  malfunction_affects_record: boolean | null;
  messages_on_this_device: boolean;
  declarant_name: string;
  declarant_position: string;
  declaration_confirmed: boolean;
}

// ── Certificate records ───────────────────────────────────────────────────

// Summary record returned by the list endpoint
export interface CertificateRecord {
  cert_id: string;
  certificate_number: string;
  document_type: "CERTIFICATE" | "DEFICIENCY_REPORT";
  outcome: CertificateOutcome;
  analysis_id: string;
  created_at: string; // ISO-8601
}

// Condition result from the gate
export interface ConditionResult {
  code: string;
  title: string;
  met: boolean;
  basis: string[];
  remedy: string | null;
}

// Full certificate record returned by GET /<id>
export interface CertificateDetail extends CertificateRecord {
  device_facts: DeviceFacts;
  evidence: Array<{
    evidence_id: string;
    type: string;
    file_name: string;
    sha256: string | null;
    server_sha256: string | null;
    hash_match: boolean | null;
    added_at: string;
  }>;
  gate: {
    document_type: "CERTIFICATE" | "DEFICIENCY_REPORT";
    outcome: CertificateOutcome;
    conditions: ConditionResult[];
    unmet: string[];
    gate_version: string;
  };
  analysis_summary: {
    overall_verdict: string;
    plain_summary: string;
    missing_information: string[];
    strengthen_checklist: string[];
    disclaimer?: string;
  };
  schema_version: number;
}
