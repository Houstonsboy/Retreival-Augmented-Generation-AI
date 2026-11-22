import { ref, uploadBytes, getDownloadURL, deleteObject } from "firebase/storage";
import {
  collection, query, where, getDocs, addDoc, deleteDoc, doc,
  orderBy, serverTimestamp, Timestamp,
} from "firebase/firestore";
import { storage, db } from "./firebase";
import { apiFetch } from "./api";

export interface EvidenceStorageRecord {
  id: string;
  sha256: string;
  fileName: string;
  fileSize: number;
  mimeType: string;
  evidenceType: string;
  storageRef: string;
  downloadUrl: string;
  caseId: string;
  caseTitle: string;
  uploadedAt: Date;
}

// ── Dedup ──────────────────────────────────────────────────────────────────
// Per-user: check if this exact file (by SHA-256) is already in Firestore.
export async function findEvidenceByHash(
  uid: string,
  sha256: string,
): Promise<EvidenceStorageRecord | null> {
  const q = query(
    collection(db, "users", uid, "evidence"),
    where("sha256", "==", sha256),
  );
  const snap = await getDocs(q);
  if (snap.empty) return null;
  const doc = snap.docs[0];
  const data = doc.data();
  return {
    id: doc.id,
    sha256: data.sha256,
    fileName: data.fileName,
    fileSize: data.fileSize,
    mimeType: data.mimeType ?? "",
    evidenceType: data.evidenceType ?? "",
    storageRef: data.storageRef,
    downloadUrl: data.downloadUrl,
    caseId: data.caseId,
    caseTitle: data.caseTitle ?? "",
    uploadedAt: data.uploadedAt instanceof Timestamp
      ? data.uploadedAt.toDate()
      : new Date(data.uploadedAt ?? 0),
  };
}

// ── Storage upload ─────────────────────────────────────────────────────────
function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout>;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(
      () => reject(new Error(`${label} timed out after ${ms / 1000}s. Go to the Firebase console → Storage and click "Get Started" to activate it on this project.`)),
      ms,
    );
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

async function uploadToStorage(
  uid: string,
  file: File,
  sha256: string,
): Promise<{ storageRef: string; downloadUrl: string }> {
  const storageRef = ref(storage, `evidence/${uid}/${sha256}/${file.name}`);
  const snap = await withTimeout(uploadBytes(storageRef, file), 20_000, "Firebase Storage upload");
  const downloadUrl = await getDownloadURL(snap.ref);
  return { storageRef: snap.ref.fullPath, downloadUrl };
}

// ── Firestore write ────────────────────────────────────────────────────────
async function saveToFirestore(
  uid: string,
  record: Omit<EvidenceStorageRecord, "id" | "uploadedAt">,
): Promise<string> {
  const docRef = await addDoc(collection(db, "users", uid, "evidence"), {
    ...record,
    uploadedAt: serverTimestamp(),
  });
  return docRef.id;
}

// ── Main upload function (dedup + upload + save) ───────────────────────────
export async function uploadEvidenceFile(
  uid: string,
  file: File,
  sha256: string,
  evidenceType: string,
  caseId: string,
  caseTitle: string,
): Promise<{ record: EvidenceStorageRecord; wasDedup: boolean }> {
  const existing = await findEvidenceByHash(uid, sha256);
  if (existing) return { record: existing, wasDedup: true };

  const { storageRef, downloadUrl } = await uploadToStorage(uid, file, sha256);

  const base: Omit<EvidenceStorageRecord, "id" | "uploadedAt"> = {
    sha256,
    fileName: file.name,
    fileSize: file.size,
    mimeType: file.type || "image/jpeg",
    evidenceType,
    storageRef,
    downloadUrl,
    caseId,
    caseTitle,
  };

  const id = await saveToFirestore(uid, base);
  return {
    record: { id, ...base, uploadedAt: new Date() },
    wasDedup: false,
  };
}

// ── Delete evidence (Firestore record + Storage file) ─────────────────────
export async function deleteEvidenceFile(
  uid: string,
  evidenceId: string,
  storagePath: string,
): Promise<void> {
  await deleteDoc(doc(db, "users", uid, "evidence", evidenceId));
  try {
    await deleteObject(ref(storage, storagePath));
  } catch {
    // Best-effort — don't fail if the Storage object is already gone
  }
}

// ── List all evidence for a user ───────────────────────────────────────────
export async function listUserEvidence(
  uid: string,
): Promise<EvidenceStorageRecord[]> {
  const q = query(
    collection(db, "users", uid, "evidence"),
    orderBy("uploadedAt", "desc"),
  );
  const snap = await getDocs(q);
  return snap.docs.map((doc) => {
    const data = doc.data();
    return {
      id: doc.id,
      sha256: data.sha256,
      fileName: data.fileName,
      fileSize: data.fileSize,
      mimeType: data.mimeType ?? "",
      evidenceType: data.evidenceType ?? "",
      storageRef: data.storageRef,
      downloadUrl: data.downloadUrl,
      caseId: data.caseId,
      caseTitle: data.caseTitle ?? "Untitled Case",
      uploadedAt: data.uploadedAt instanceof Timestamp
        ? data.uploadedAt.toDate()
        : new Date(data.uploadedAt ?? 0),
    };
  });
}

// ── LLM case title ─────────────────────────────────────────────────────────
export async function generateCaseTitle(userAccount: string): Promise<string> {
  if (!userAccount.trim()) return "Untitled Case";
  try {
    const res = await apiFetch("/sccvision/case/title", {
      method: "POST",
      body: JSON.stringify({ user_account: userAccount }),
    });
    if (!res.ok) return "Untitled Case";
    const { title } = await res.json();
    return (title as string)?.trim() || "Untitled Case";
  } catch {
    return "Untitled Case";
  }
}
