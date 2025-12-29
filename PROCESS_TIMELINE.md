# Process Timeline: Extract Embeddings Flow

This document traces the complete flow from clicking "Process Documents for Chat" in the frontend to embedding documents into the vector database.

## Overview
- **Frontend Button**: "Process Documents for Chat" (in `/frontend/app/ingester/page.tsx`)
- **Backend Endpoint**: `/api/digest` (in `/backend/app.py`)
- **FIRAC Extraction**: `firac.py` → `run_firac_from_file()`
- **Ingestion**: `ingester.py` → `ingest_firac_data()`

---

## Step-by-Step Process Flow

### 1. Frontend: User Clicks Button
**File**: `frontend/app/ingester/page.tsx`
**Function**: `handleDigest()` (lines 121-150)

```typescript
const handleDigest = async () => {
  // Sets loading state
  setIsDigesting(true);
  
  // Sends POST request to backend
  const response = await fetch("http://localhost:5000/api/digest", {
    method: "POST",
    headers: { "Content-Type": "application/json" }
  });
  
  // Handles response
  const data = await response.json();
  // Shows success/error message
}
```

**What happens**: 
- Frontend sends POST request to `http://localhost:5000/api/digest`
- No request body (empty POST)
- Waits for response with processing results

---

### 2. Backend: Digest Endpoint Receives Request
**File**: `backend/app.py`
**Function**: `digest()` (lines 151-266)
**Route**: `@app.route('/api/digest', methods=['POST'])`

**Step 2.1: Setup**
```python
# Captures stdout for logging
sys.stdout = captured_output = io.StringIO()

# Checks if directory exists
RES_IPSA_LOQUITUR_DIR = REPO_DIR / "Res_ipsa_loquitur"
if not RES_IPSA_LOQUITUR_DIR.exists():
    return error 404
```

**Step 2.2: Get All PDF Files**
```python
# Line 174: Gets ALL PDFs from Res_ipsa_loquitur directory
pdf_files = list(RES_IPSA_LOQUITUR_DIR.glob("*.pdf"))

# Expected: 17 PDF files (based on directory listing)
```

**Step 2.3: Initialize Counters**
```python
files_processed = 0  # Total files attempted
files_succeeded = 0   # Successfully embedded
files_failed = 0      # Failed during processing
failed_files = []     # List of failed files with errors
```

**Step 2.4: Process Each PDF File** (Loop starts at line 196)
```python
for pdf_file in pdf_files:  # Iterates through all 17 PDFs
    print(f"Processing: {pdf_file.name}")
    
    try:
        # STEP A: Extract FIRAC components
        firac_result = run_firac_from_file(pdf_file)
        
        # STEP B: Check for FIRAC extraction errors
        if firac_result.get('error'):
            files_failed += 1
            failed_files.append({'file': pdf_file.name, 'error': firac_result['error']})
            continue  # Skip to next file
        
        # STEP C: Ingest into ChromaDB
        ingest_firac_data(
            firac_data=firac_result,
            source_file_path=pdf_file
        )
        
        # STEP D: Success
        files_succeeded += 1
        files_processed += 1
        
    except Exception as e:
        # STEP E: Catch any exceptions during processing
        files_failed += 1
        failed_files.append({'file': pdf_file.name, 'error': str(e)})
        continue  # Skip to next file
```

**Step 2.5: Return Response**
```python
return jsonify({
    'message': f'Processed {files_processed} file(s): {files_succeeded} succeeded, {files_failed} failed',
    'files_processed': files_processed,
    'files_succeeded': files_succeeded,
    'files_failed': files_failed,
    'failed_files': failed_files,  # List of files that failed
    'output': output,  # Captured stdout logs
    'status': 'success' if files_succeeded > 0 else 'partial'
})
```

---

### 3. FIRAC Extraction: `run_firac_from_file()`
**File**: `backend/firac.py`
**Function**: `run_firac_from_file(file_path: Path)` (lines 1815-1907)

**Step 3.1: Initialize Result Dictionary**
```python
result = {
    'document': '',      # Full document text
    'metadata': '',       # Case metadata
    'facts': '',         # Extracted facts
    'issues': '',        # Extracted issues
    'rules': '',         # Extracted rules
    'application': '',   # Extracted application
    'conclusion': '',    # Extracted conclusion
    'error': None        # Error message if extraction fails
}
```

**Step 3.2: Load PDF Document**
```python
# Line 1851: Loads PDF and extracts text (with OCR fallback)
document_text = load_document_from_path(file_path)

# If loading fails, returns error string starting with "Error:"
if document_text.startswith("Error:"):
    result['error'] = document_text
    return result  # Returns with error, no FIRAC extraction
```

**Step 3.3: Extract FIRAC Components**
```python
# Line 1862: Single API call to Groq LLM to extract all FIRAC components
extraction_result = extract_all_firac_combined(document_text)

# Parses response into sections:
result['facts'] = extraction_result.get('facts', '')
result['issues'] = extraction_result.get('issues', '')
result['rules'] = extraction_result.get('rules', '')
result['application'] = extraction_result.get('application', '')
result['conclusion'] = extraction_result.get('conclusion', '')
```

**Step 3.4: Extract Metadata**
```python
# Line 1873: Separate API call to extract case metadata
metadata_text = extract_metadata_only(document_text)
result['metadata'] = metadata_text
```

**Step 3.5: Return Result**
```python
# Returns dictionary with FIRAC components and metadata
# If any step fails, result['error'] will contain error message
return result
```

**Possible Failure Points**:
1. PDF file cannot be read → `document_text.startswith("Error:")`
2. Groq API call fails → Exception caught, `result['error']` set
3. FIRAC extraction returns empty sections → No error, but empty content

---

### 4. Ingestion: `ingest_firac_data()`
**File**: `backend/ingester.py`
**Function**: `ingest_firac_data(firac_data, case_identifier, source_file_path)` (lines 141-324)

**Step 4.1: Error Check**
```python
# Line 185: Check if FIRAC extraction had errors
if firac_data.get("error"):
    raise ValueError(f"Cannot ingest FIRAC data: {firac_data['error']}")
    # This exception is caught in app.py, file marked as failed
```

**Step 4.2: Initialize ChromaDB**
```python
# Line 192: Connect to ChromaDB
client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
collection = client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)
```

**Step 4.3: Parse Metadata**
```python
# Line 204: Parse metadata text into structured dictionary
parsed_metadata = parse_metadata(metadata_text)

# Extract case identifier (used as unique key)
case_identifier = parsed_metadata.get("file_name", "unknown_case")
```

**Step 4.4: Prepare Chunks**
```python
# Lines 234-275: For each FIRAC component (facts, issues, rules, application, conclusion)
for component in FIRAC_COMPONENTS:
    content = firac_data.get(component, "").strip()
    
    # Skip empty components
    if not content:
        print(f"⚠ Skipping empty {component} component")
        continue  # This component not added to chunks
    
    # Create chunk metadata
    chunk_metadata = {
        "firac_component": component,
        "case_identifier": case_identifier,
        "file_name": parsed_metadata.get("file_name", ""),
        # ... other metadata fields
    }
    
    # Generate unique chunk ID
    chunk_id = f"{safe_case_id}_{component}_{timestamp}"
    
    chunks.append(content)
    chunk_metadatas.append(chunk_metadata)
    chunk_ids.append(chunk_id)
```

**Step 4.5: Check for Existing Chunks**
```python
# Lines 290-297: Check if case already exists in database
existing = collection.get(
    where={"case_identifier": case_identifier},
    include=["metadatas"]
)

# If exists, DELETE old chunks before adding new ones
if existing.get("ids"):
    print(f"⚠ Found {len(existing['ids'])} existing chunk(s). Removing...")
    collection.delete(where={"case_identifier": case_identifier})
```

**Step 4.6: Create Embeddings**
```python
# Line 283: Create embeddings for all chunks
embeddings = create_embeddings(chunks, model)

# Uses SentenceTransformer model: "intfloat/e5-base-v2"
# Each chunk is prefixed with "passage: " before encoding
```

**Step 4.7: Store in ChromaDB**
```python
# Lines 301-306: Store chunks with embeddings and metadata
collection.add(
    ids=chunk_ids,
    documents=chunks,
    embeddings=embeddings.tolist(),
    metadatas=chunk_metadatas,
)
```

**Possible Failure Points**:
1. FIRAC data has error → Raises ValueError (caught in app.py)
2. All FIRAC components are empty → No chunks to store (but no error raised!)
3. ChromaDB connection fails → Exception (caught in app.py)
4. Embedding creation fails → Exception (caught in app.py)

---

## Critical Issue: Empty FIRAC Components

**Problem**: If FIRAC extraction succeeds but returns empty components, `ingest_firac_data()` will:
- Skip empty components (line 238-240)
- If ALL components are empty, no chunks are created
- Function returns without error (line 279: just prints warning)
- File is marked as "succeeded" in app.py (line 221)
- **But nothing is actually stored in the database!**

**This explains why only 2 documents are embedded despite 17 being processed.**

---

## Debugging Checklist

To identify why documents aren't being embedded:

1. **Check the API response**:
   - Look at `failed_files` array in the response
   - Check `files_succeeded` vs `files_processed`
   - Review `output` field for detailed logs

2. **Check FIRAC extraction results**:
   - Look for files where FIRAC extraction returned empty components
   - Check if Groq API calls are timing out or failing
   - Verify PDF text extraction is working

3. **Check ingestion logs**:
   - Look for "⚠ Skipping empty {component} component" messages
   - Check if "No valid FIRAC components to ingest" appears
   - Verify case identifiers are being extracted correctly

4. **Check ChromaDB**:
   - Verify chunks are actually being stored
   - Check if duplicate case identifiers are overwriting each other
   - Verify embeddings are being created

---

## Expected Behavior

**For each of 17 PDF files**:
1. ✅ PDF loaded and text extracted
2. ✅ FIRAC components extracted via Groq API
3. ✅ Metadata extracted
4. ✅ 5 chunks created (one per FIRAC component)
5. ✅ Embeddings generated
6. ✅ Chunks stored in ChromaDB

**Result**: 17 cases × 5 components = 85 chunks in database

**Current State**: Only 2 cases × 5 components = 10 chunks in database

---

## Next Steps to Fix

1. **Add validation in `ingest_firac_data()`**:
   - Check if any chunks were created before marking as success
   - Raise error if all components are empty

2. **Add logging in `app.py`**:
   - Log which files succeed vs fail
   - Log FIRAC extraction results for each file
   - Log ingestion results

3. **Check Groq API responses**:
   - Verify API calls are succeeding
   - Check if responses contain valid FIRAC sections
   - Monitor for rate limiting or timeout errors

4. **Verify PDF text extraction**:
   - Ensure PDFs can be read
   - Check if OCR is needed for some files
   - Verify text extraction quality

