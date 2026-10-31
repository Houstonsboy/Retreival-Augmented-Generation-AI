from flask import Flask, request, jsonify, send_from_directory, send_file, g
from flask_cors import CORS
import re
import time
import os
import sys
import io
import json
import requests
import uuid as uuid_mod
import urllib.parse as urllib_parse
from auth import init_firebase_admin, require_role
from firebase_admin import firestore as fb_firestore, storage as fb_storage
from pathlib import Path
from werkzeug.utils import secure_filename
from digester import digest_repo
from pixe import run_rag_pipeline
from CaseLaw.firac import run_firac, run_firac_from_file
from CaseLaw.ingester import ingest_firac_data
from Qclassifier import Qclassifier
from RulesRetriever import retrieve_relevant_rules
from constitutionlogic.constitutionbreaker import extract_constitution_articles, OUTPUT_DIR as CONST_OUTPUT_DIR, flatten_articles_to_embeddable_chunks, CHUNK_SIZE, CHUNK_OVERLAP
from constitutionlogic.constitution_digester import ConstitutionDigester, EMBEDDABLE_CHUNKS_JSON
from config import CASELAW_DB_DIR, CASELAW_COLLECTION_NAME, SCC_CASES_DIR
app = Flask(__name__)
_CORS_ORIGIN = os.environ.get("CORS_ORIGIN", "http://localhost:3000")
CORS(app, origins=_CORS_ORIGIN)
init_firebase_admin()

# Path to the Repo directory
REPO_DIR = Path(__file__).resolve().parent / "Repo"
RES_IPSA_LOQUITUR_DIR = REPO_DIR / "Res_ipsa_loquitur"

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/api/chat', methods=['POST'])
@require_role("advocate", "admin")
def chat():
    """
    RAG endpoint that processes queries using the pixe.py pipeline.
    Retrieves relevant information from ChromaDB and generates responses using Groq.
    """
    data = request.get_json()
    query = data.get('message', '')
    
    if not query:
        return jsonify({'error': 'No message provided'}), 400
    
    if not query.strip():
        return jsonify({'error': 'Query cannot be empty'}), 400
    
    try:
        # Capture stdout to get pipeline output (optional, for debugging)
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            # Run the RAG pipeline
            response_text = run_rag_pipeline(query)
            output = captured_output.getvalue()
            
            # Return the response
            return jsonify({
                'response': response_text,
                'status': 'success'
            }), 200
            
        except RuntimeError as e:
            # Handle cases where ChromaDB is not initialized
            error_msg = str(e)
            if "ChromaDB" in error_msg or "empty" in error_msg.lower():
                return jsonify({
                    'error': 'No documents have been processed yet. Please upload and digest documents first.',
                    'response': 'I need documents to be processed before I can answer questions. Please use the Ingester page to upload documents and run the digest process.',
                    'status': 'error'
                }), 503
            else:
                return jsonify({
                    'error': f'RAG pipeline error: {error_msg}',
                    'response': 'Sorry, I encountered an error processing your query. Please try again.',
                    'status': 'error'
                }), 500
                
        except Exception as e:
            output = captured_output.getvalue()
            return jsonify({
                'error': f'Unexpected error: {str(e)}',
                'response': 'Sorry, I encountered an unexpected error. Please try again.',
                'status': 'error'
            }), 500
        finally:
            sys.stdout = old_stdout
            
    except Exception as e:
        return jsonify({
            'error': f'Failed to process query: {str(e)}',
            'response': 'Sorry, I encountered an error. Please try again.',
            'status': 'error'
        }), 500

@app.route('/api/files', methods=['GET'])
@require_role("advocate", "admin")
def get_files():
    """
    Return list of files in the Repo directory.
    """
    if not REPO_DIR.exists():
        return jsonify({'files': []})
    
    files = [f.name for f in REPO_DIR.iterdir() if f.is_file()]
    return jsonify({'files': files})

@app.route('/api/upload', methods=['POST'])
@require_role("admin")
def upload_file():
    """
    Handle file uploads and save them to the Repo directory.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file is a PDF
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    # Ensure Repo directory exists
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Secure the filename to prevent directory traversal
    filename = secure_filename(file.filename)
    file_path = REPO_DIR / filename
    
    # Save the file
    try:
        file.save(str(file_path))
        return jsonify({'message': 'File uploaded successfully', 'filename': filename}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500

@app.route('/api/files/<filename>', methods=['GET'])
@require_role("advocate", "admin")
def serve_file(filename):
    """
    Serve PDF files from the Repo directory.
    """
    # Secure the filename to prevent directory traversal
    safe_filename = secure_filename(filename)
    file_path = REPO_DIR / safe_filename
    
    # Check if file exists and is a PDF
    if not file_path.exists():
        return jsonify({'error': 'File not found'}), 404
    
    if not safe_filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files can be served'}), 400
    
    # Serve the file
    return send_from_directory(
        str(REPO_DIR),
        safe_filename,
        mimetype='application/pdf',
        as_attachment=False
    )

@app.route('/api/digest', methods=['POST'])
@require_role("admin")
def digest():
    """
    Process PDF files: Extract FIRAC and embed into ChromaDB.
    Outputs processing steps directly to the server console.
    """
    try:
        # --- INPUT VALIDATION & INITIALIZATION ---
        data = request.get_json() or {}
        file_path = data.get('file_path')
        
        if file_path:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return jsonify({'error': f'File not found: {file_path}', 'status': 'error'}), 404
            pdf_files = [file_path_obj]
        else:
            if not RES_IPSA_LOQUITUR_DIR.exists():
                return jsonify({'error': f'Directory not found: {RES_IPSA_LOQUITUR_DIR}', 'status': 'error'}), 404
            pdf_files = list(RES_IPSA_LOQUITUR_DIR.glob("*.pdf"))

        if not pdf_files:
            print("\n[!] No PDF files found to process.")
            return jsonify({'message': 'No files found', 'status': 'success'}), 200

        # --- LOGGING START ---
        print(f"\n{'='*80}")
        print(f"🚀 STARTING DIGEST PROCESS")
        print(f"Target: {'Specific file' if file_path else 'Directory scan'}")
        print(f"Total files to process: {len(pdf_files)}")
        print(f"{'='*80}\n")
        
        files_processed = 0
        files_succeeded = 0
        files_failed = 0
        failed_files = []
        
        for idx, pdf_file in enumerate(pdf_files, 1):
            print(f"({idx}/{len(pdf_files)}) 📂 FILE: {pdf_file.name}")
            print(f"{'-'*40}")
            
            try:
                # STEP 1: FIRAC EXTRACTION
                print(f"  [STEP 1] Extracting FIRAC via LLM...")
                firac_result = run_firac_from_file(pdf_file)
                
                if firac_result.get('error'):
                    print(f"  [❌] Extraction Error: {firac_result['error']}")
                    files_failed += 1
                    failed_files.append({'file': pdf_file.name, 'error': firac_result['error']})
                    continue
                
                # Validation of components
                firac_components = ['facts', 'issues', 'rules', 'application', 'conclusion']
                empty = [c for c in firac_components if not firac_result.get(c, '').strip()]
                if empty:
                    print(f"  [!] Warning: Missing data in fields: {', '.join(empty)}")
                
                # STEP 2: CHROMA DB INGESTION
                print(f"  [STEP 2] Generating embeddings and saving to ChromaDB...")
                ingest_firac_data(
                    firac_data=firac_result,
                    source_file_path=pdf_file
                )
                
                files_succeeded += 1
                files_processed += 1
                print(f"  [✅] SUCCESS: {pdf_file.name} is now searchable.\n")
                
            except Exception as e:
                print(f"  [❌] CRITICAL ERROR processing {pdf_file.name}: {str(e)}")
                files_failed += 1
                failed_files.append({'file': pdf_file.name, 'error': str(e)})
                continue

        # --- FINAL SUMMARY ---
        print(f"{'='*80}")
        print(f"🏁 PROCESS COMPLETE")
        print(f"   - Total Succeeded: {files_succeeded}")
        print(f"   - Total Failed:    {files_failed}")
        print(f"{'='*80}\n")
        
        return jsonify({
            'message': f'Processed {files_processed} file(s)',
            'files_succeeded': files_succeeded,
            'files_failed': files_failed,
            'failed_files': failed_files,
            'status': 'success' if files_succeeded > 0 else 'failed'
        }), 200

    except Exception as e:
        print(f"\n[🚨] GLOBAL API ERROR: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}', 'status': 'error'}), 500

@app.route('/api/firac', methods=['POST'])
@require_role("advocate", "admin")
def firac():
    """
    Execute firac.py to process a PDF document (defaults to Wilson Wanjala).
    Accepts optional file_path in request body to process a specific file.
    Returns the extracted and cleaned text content.
    
    Request body (optional):
    {
        "file_path": "path/to/file.pdf"  # If not provided, uses Wilson Wanjala PDF
    }
    """
    try:
        data = request.get_json() or {}
        file_path = data.get('file_path')
        
        # Capture stdout to get processing output (logs/print statements)
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            # If file_path is provided, use it; otherwise use default Wilson PDF
            if file_path:
                file_path_obj = Path(file_path)
                if not file_path_obj.exists():
                    return jsonify({
                        'error': f'File not found: {file_path}',
                        'status': 'error'
                    }), 404
                # Process specific file using combined API call
                result = run_firac_from_file(file_path_obj)
            else:
                # Default: Process Wilson Wanjala PDF using combined API call
                result = run_firac()
            
            output = captured_output.getvalue()
            
            if result.get('error'):
                # Return error response with same structure for consistency
                return jsonify({
                    'document': result.get('document', ''),
                    'metadata': result.get('metadata', ''),
                    'facts': {
                        'content': result.get('facts', ''),
                        'metadata': result.get('facts_metadata', '')
                    },
                    'issues': {
                        'content': result.get('issues', ''),
                        'metadata': result.get('issues_metadata', '')
                    },
                    'rules': {
                        'content': result.get('rules', ''),
                        'metadata': result.get('rules_metadata', '')
                    },
                    'application': {
                        'content': result.get('application', ''),
                        'metadata': result.get('application_metadata', '')
                    },
                    'conclusion': {
                        'content': result.get('conclusion', ''),
                        'metadata': result.get('conclusion_metadata', '')
                    },
                    'full_response': result.get('full_response', ''),
                    'output': output,
                    'error': result.get('error'),
                    'status': 'error'
                }), 500
            
            return jsonify({
                'document': result.get('document', ''),
                'metadata': result.get('metadata', ''),
                'facts': {
                    'content': result.get('facts', ''),
                    'metadata': result.get('facts_metadata', '')
                },
                'issues': {
                    'content': result.get('issues', ''),
                    'metadata': result.get('issues_metadata', '')
                },
                'rules': {
                    'content': result.get('rules', ''),
                    'metadata': result.get('rules_metadata', '')
                },
                'application': {
                    'content': result.get('application', ''),
                    'metadata': result.get('application_metadata', '')
                },
                'conclusion': {
                    'content': result.get('conclusion', ''),
                    'metadata': result.get('conclusion_metadata', '')
                },
                'full_response': result.get('full_response', ''),
                'output': output,
                'status': 'success'
            }), 200 
            
        except FileNotFoundError as e:
            output = captured_output.getvalue()
            file_name = file_path if file_path else "Wilson Wanjala PDF"
            return jsonify({
                'error': f'PDF not found: {str(e)}',
                'content': f'Error: {file_name} file not found in the repository.',
                'output': output,
                'status': 'error'
            }), 404
        except Exception as e:
            output = captured_output.getvalue()
            return jsonify({
                'error': f'FIRAC processing failed: {str(e)}',
                'content': f'Error processing document: {str(e)}',
                'output': output,
                'status': 'error'
            }), 500
        finally:
            sys.stdout = old_stdout
            
    except Exception as e:
            return jsonify({
                'error': f'Unexpected error: {str(e)}',
                'content': f'Unexpected error: {str(e)}',
                'status': 'error'
            }), 500

@app.route('/api/ingest-firac', methods=['POST'])
@require_role("admin")
def ingest_firac():
    """
    Ingest FIRAC data into ChromaDB vector database.
    Accepts a file path, extracts FIRAC using firac.py, then stores in ChromaDB.
    
    Request body (optional):
    {
        "file_path": "path/to/file.pdf"  # If not provided, uses frontend data (backward compatibility)
    }
    """
    try:
        data = request.get_json() or {}
        
        # Capture stdout to get ingestion output
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            # If file_path is provided, extract FIRAC from that file
            if data.get('file_path'):
                file_path = Path(data['file_path'])
                
                if not file_path.exists():
                    return jsonify({
                        'error': f'File not found: {file_path}',
                        'status': 'error'
                    }), 404
                
                # Extract FIRAC from file
                firac_result = run_firac_from_file(file_path)
                
                if firac_result.get('error'):
                    output = captured_output.getvalue()
                    return jsonify({
                        'error': f'FIRAC extraction failed: {firac_result["error"]}',
                        'output': output,
                        'status': 'error'
                    }), 400
                
                # Ingest into ChromaDB
                ingest_firac_data(
                    firac_data=firac_result,
                    source_file_path=file_path
                )
                
                output = captured_output.getvalue()
                return jsonify({
                    'message': f'FIRAC data extracted and ingested successfully for {file_path.name}',
                    'output': output,
                    'status': 'success'
                }), 200
            
            # Backward compatibility: accept FIRAC data from frontend
            if not data:
                return jsonify({
                    'error': 'No file path or FIRAC data provided',
                    'status': 'error'
                }), 400
            
            # Reconstruct FIRAC data structure from frontend format (backward compatibility)
            firac_data = {
                'document': data.get('document', ''),
                'metadata': data.get('metadata', ''),
                'facts': data.get('facts', {}).get('content', '') if isinstance(data.get('facts'), dict) else data.get('facts', ''),
                'issues': data.get('issues', {}).get('content', '') if isinstance(data.get('issues'), dict) else data.get('issues', ''),
                'rules': data.get('rules', {}).get('content', '') if isinstance(data.get('rules'), dict) else data.get('rules', ''),
                'application': data.get('application', {}).get('content', '') if isinstance(data.get('application'), dict) else data.get('application', ''),
                'conclusion': data.get('conclusion', {}).get('content', '') if isinstance(data.get('conclusion'), dict) else data.get('conclusion', ''),
                'facts_metadata': data.get('facts', {}).get('metadata', '') if isinstance(data.get('facts'), dict) else data.get('metadata', ''),
                'issues_metadata': data.get('issues', {}).get('metadata', '') if isinstance(data.get('issues'), dict) else data.get('metadata', ''),
                'rules_metadata': data.get('rules', {}).get('metadata', '') if isinstance(data.get('rules'), dict) else data.get('metadata', ''),
                'application_metadata': data.get('application', {}).get('metadata', '') if isinstance(data.get('application'), dict) else data.get('metadata', ''),
                'conclusion_metadata': data.get('conclusion', {}).get('metadata', '') if isinstance(data.get('conclusion'), dict) else data.get('metadata', ''),
                'error': None
            }
            
            source_file_path = data.get('source_file_path')
            case_identifier = data.get('case_identifier')
            
            # Ingest the FIRAC data
            ingest_firac_data(
                firac_data=firac_data,
                case_identifier=case_identifier,
                source_file_path=Path(source_file_path) if source_file_path else None
            )
            
            output = captured_output.getvalue()
            
            return jsonify({
                'message': 'FIRAC data ingested successfully into vector database',
                'output': output,
                'status': 'success'
            }), 200
            
        except ValueError as e:
            output = captured_output.getvalue()
            return jsonify({
                'error': f'Invalid FIRAC data: {str(e)}',
                'output': output,
                'status': 'error'
            }), 400
        except Exception as e:
            output = captured_output.getvalue()
            return jsonify({
                'error': f'Ingestion failed: {str(e)}',
                'output': output,
                'status': 'error'
            }), 500
        finally:
            sys.stdout = old_stdout
            
    except Exception as e:
        return jsonify({
            'error': f'Unexpected error: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/digest/constitution', methods=['POST'])
@require_role("admin")
def digest_constitution():
    """
    Process Constitution PDF: Extract articles and embed into ChromaDB.
    
    Request body (optional):
    {
        "pdf_path": "/path/to/constitution.pdf",  // Optional: specific PDF to process
        "force_reextract": false,                 // Optional: force re-extraction even if JSON exists
        "skip_existing": true                     // Optional: skip chunks already in DB
    }
    """
    try:
        # --- INPUT VALIDATION ---
        data = request.get_json() or {}
        pdf_path = data.get('pdf_path')
        force_reextract = data.get('force_reextract', False)
        skip_existing = data.get('skip_existing', True)
        
        # --- LOGGING START ---
        print(f"\n{'='*80}")
        print(f"🏛️  CONSTITUTION DIGEST PROCESS STARTED")
        print(f"{'='*80}")
        print(f"⚙️  Configuration:")
        print(f"    • Custom PDF: {pdf_path if pdf_path else 'Using default Constitution PDF'}")
        print(f"    • Force re-extraction: {force_reextract}")
        print(f"    • Skip existing chunks: {skip_existing}")
        print(f"{'='*80}\n")
        
        extraction_stats = {}
        ingestion_stats = {}
        
        # ============================================================
        # STEP 1: ARTICLE EXTRACTION
        # ============================================================
        print(f"{'─'*80}")
        print(f"📖 STEP 1: EXTRACTING CONSTITUTION ARTICLES")
        print(f"{'─'*80}\n")
        
        embeddable_json_path = CONST_OUTPUT_DIR / "constitution_embeddable_chunks.json"
        
        # Check if extraction is needed
        should_extract = force_reextract or not embeddable_json_path.exists()
        
        if should_extract:
            print(f"🔧 Running article extraction from PDF...")
            
            try:
                # Run extraction (constitutionbreaker.py main logic)
                if pdf_path:
                    print(f"   📄 Source: {pdf_path}\n")
                    articles = extract_constitution_articles(pdf_path=Path(pdf_path))
                else:
                    print(f"   📄 Source: Default Constitution PDF\n")
                    articles = extract_constitution_articles()
                
                if not articles:
                    print(f"❌ ERROR: No articles extracted from PDF\n")
                    return jsonify({
                        'error': 'Article extraction returned no results',
                        'status': 'error'
                    }), 500
                
                # Create embeddable chunks JSON file
                print(f"📝 Creating embeddable chunks JSON...")
                embeddable_chunks = flatten_articles_to_embeddable_chunks(articles)
                
                # Ensure output directory exists
                CONST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                
                # Create JSON data structure
                json_data = {
                    "metadata": {
                        "total_articles": len(articles),
                        "total_embeddable_chunks": len(embeddable_chunks),
                        "chunk_size": CHUNK_SIZE,
                        "chunk_overlap": CHUNK_OVERLAP,
                        "source_document": articles[0]['source_document'] if articles else None,
                        "description": "Each chunk is an individual embeddable unit. 'chunk_text' contains either the full article (for short articles) or the specific chunk (for long articles). Embed 'chunk_text' and store all metadata."
                    },
                    "embeddable_chunks": embeddable_chunks
                }
                
                # Write JSON file
                with open(embeddable_json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                
                print(f"✅ EXTRACTION COMPLETE")
                print(f"   • Total articles extracted: {len(articles)}")
                print(f"   • Total embeddable chunks: {len(embeddable_chunks)}")
                print(f"   • JSON saved to: {embeddable_json_path.name}\n")
                
            except Exception as e:
                print(f"❌ EXTRACTION FAILED: {str(e)}\n")
                return jsonify({
                    'error': f'Article extraction failed: {str(e)}',
                    'status': 'error'
                }), 500
        else:
            print(f"⏭️  SKIPPING EXTRACTION - Using cached JSON")
            print(f"   • File: {embeddable_json_path.name}")
            print(f"   • Tip: Set force_reextract=true to regenerate\n")
        
        # Load extraction metadata
        try:
            with open(embeddable_json_path, 'r', encoding='utf-8') as f:
                extraction_data = json.load(f)
                extraction_stats = extraction_data.get('metadata', {})
                chunks = extraction_data.get('embeddable_chunks', [])
                
            print(f"📊 JSON Summary:")
            print(f"   • Total articles: {extraction_stats.get('total_articles', 'N/A')}")
            print(f"   • Total embeddable chunks: {len(chunks)}")
            print(f"   • Source document: {extraction_stats.get('source_document', 'N/A')}")
            print(f"   • Chunk size: {extraction_stats.get('chunk_size', 'N/A')} chars")
            print(f"   • Chunk overlap: {extraction_stats.get('chunk_overlap', 'N/A')} chars\n")
            
        except Exception as e:
            print(f"⚠️  WARNING: Could not read extraction metadata: {str(e)}\n")
            return jsonify({
                'error': f'Failed to read JSON file: {str(e)}',
                'status': 'error'
            }), 500
        
        # ============================================================
        # STEP 2: CHROMADB INGESTION WITH DETAILED LOGGING
        # ============================================================
        print(f"{'─'*80}")
        print(f"💾 STEP 2: EMBEDDING INTO CHROMADB")
        print(f"{'─'*80}\n")
        
        try:
            # Initialize digester
            digester = ConstitutionDigester()
            
            print(f"🚀 Starting embedding process...\n")
            
            # Track statistics
            stats = {
                'total_chunks': len(chunks),
                'ingested': 0,
                'skipped': 0,
                'failed': 0,
                'failed_chunks': [],
                'articles_processed': set(),  # Track unique articles
                'article_details': []  # Store per-article results
            }
            
            # Group chunks by article for cleaner logging
            articles_map = {}
            for chunk in chunks:
                article_num = chunk['article_number']
                if article_num not in articles_map:
                    articles_map[article_num] = []
                articles_map[article_num].append(chunk)
            
            print(f"📋 Processing {len(articles_map)} unique articles...\n")
            print(f"{'─'*80}\n")
            
            # Process each article
            for idx, (article_num, article_chunks) in enumerate(sorted(articles_map.items(), key=lambda x: int(x[0])), 1):
                article_header = article_chunks[0].get('article_header', 'No header')
                total_chunks = len(article_chunks)
                
                print(f"[{idx}/{len(articles_map)}] Article {article_num}: {article_header}")
                print(f"        Chunks: {total_chunks} | ", end='', flush=True)
                
                article_ingested = 0
                article_skipped = 0
                article_failed = 0
                
                # Process each chunk in the article
                for chunk_data in article_chunks:
                    try:
                        # Generate unique ID
                        chunk_id = digester.generate_chunk_id(chunk_data)
                        
                        # Check if exists
                        if skip_existing and digester.check_existing_chunk(chunk_id):
                            stats['skipped'] += 1
                            article_skipped += 1
                            continue
                        
                        # Generate embedding
                        chunk_text = chunk_data['chunk_text']
                        embedding = digester.embed_chunk(chunk_text)
                        
                        # Prepare metadata
                        metadata = digester.prepare_metadata(chunk_data)
                        
                        # Ingest into ChromaDB
                        digester.collection.add(
                            ids=[chunk_id],
                            embeddings=[embedding],
                            documents=[chunk_text],
                            metadatas=[metadata]
                        )
                        
                        stats['ingested'] += 1
                        article_ingested += 1
                        stats['articles_processed'].add(article_num)
                        
                    except Exception as e:
                        stats['failed'] += 1
                        article_failed += 1
                        stats['failed_chunks'].append({
                            'article': article_num,
                            'chunk': chunk_data.get('chunk_index', 'unknown'),
                            'error': str(e)
                        })
                
                # Print article result
                status_parts = []
                if article_ingested > 0:
                    status_parts.append(f"✅ {article_ingested} embedded")
                if article_skipped > 0:
                    status_parts.append(f"⏭️ {article_skipped} skipped")
                if article_failed > 0:
                    status_parts.append(f"❌ {article_failed} failed")
                
                print(" | ".join(status_parts) if status_parts else "No action taken")
                
                # Store article details
                stats['article_details'].append({
                    'article_number': article_num,
                    'article_header': article_header,
                    'total_chunks': total_chunks,
                    'ingested': article_ingested,
                    'skipped': article_skipped,
                    'failed': article_failed
                })
            
            print(f"\n{'─'*80}\n")
            
            # Convert set to count
            total_articles_processed = len(stats['articles_processed'])
            
            # Final ingestion summary
            print(f"✅ INGESTION COMPLETE\n")
            print(f"📊 Final Statistics:")
            print(f"   • Unique articles processed: {total_articles_processed}")
            print(f"   • Total chunks processed: {stats['total_chunks']}")
            print(f"   • Successfully embedded: {stats['ingested']} ✅")
            print(f"   • Skipped (already exist): {stats['skipped']} ⏭️")
            print(f"   • Failed: {stats['failed']} ❌")
            
            if stats['failed'] > 0:
                print(f"\n⚠️  Failed chunks:")
                for failed in stats['failed_chunks'][:5]:  # Show first 5
                    print(f"      • Article {failed['article']}, Chunk {failed['chunk']}: {failed['error']}")
                if len(stats['failed_chunks']) > 5:
                    print(f"      ... and {len(stats['failed_chunks']) - 5} more")
            
            print(f"\n{'─'*80}\n")
            
            ingestion_stats = stats
                
        except Exception as e:
            print(f"❌ INGESTION FAILED: {str(e)}\n")
            import traceback
            traceback.print_exc()
            return jsonify({
                'error': f'ChromaDB ingestion failed: {str(e)}',
                'status': 'error',
                'extraction': extraction_stats
            }), 500
        
        # ============================================================
        # FINAL SUMMARY
        # ============================================================
        print(f"{'='*80}")
        print(f"🏁 CONSTITUTION DIGEST PROCESS COMPLETE")
        print(f"{'='*80}")
        print(f"📖 Extraction:")
        print(f"   • Total articles: {extraction_stats.get('total_articles', 'N/A')}")
        print(f"   • Total chunks: {extraction_stats.get('total_embeddable_chunks', 'N/A')}")
        print(f"   • Source: {extraction_stats.get('source_document', 'N/A')}")
        print(f"\n💾 Ingestion:")
        print(f"   • Articles processed: {len(ingestion_stats['articles_processed'])}")
        print(f"   • Chunks embedded: {ingestion_stats['ingested']} ✅")
        print(f"   • Chunks skipped: {ingestion_stats['skipped']} ⏭️")
        print(f"   • Chunks failed: {ingestion_stats['failed']} ❌")
        print(f"{'='*80}\n")
        
        # Determine overall status
        overall_status = 'success'
        if ingestion_stats['failed'] > 0:
            overall_status = 'partial_success'
        if ingestion_stats['ingested'] == 0 and ingestion_stats['skipped'] == 0:
            overall_status = 'failed'
        
        return jsonify({
            'message': 'Constitution digest process completed',
            'status': overall_status,
            'extraction': {
                'total_articles': extraction_stats.get('total_articles'),
                'total_chunks': extraction_stats.get('total_embeddable_chunks'),
                'source_document': extraction_stats.get('source_document'),
                'chunk_size': extraction_stats.get('chunk_size'),
                'chunk_overlap': extraction_stats.get('chunk_overlap')
            },
            'ingestion': {
                'articles_processed': len(ingestion_stats['articles_processed']),
                'total_chunks': ingestion_stats['total_chunks'],
                'ingested': ingestion_stats['ingested'],
                'skipped': ingestion_stats['skipped'],
                'failed': ingestion_stats['failed'],
                'failed_chunks': ingestion_stats['failed_chunks'][:10]  # Limit to first 10
            }
        }), 200
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"🚨 CRITICAL ERROR")
        print(f"{'='*80}")
        print(f"Error: {str(e)}\n")
        import traceback
        traceback.print_exc()
        print(f"{'='*80}\n")
        return jsonify({
            'error': f'Processing failed: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/digest/constitution/status', methods=['GET'])
def constitution_status():
    """
    Check the status of constitution extraction and ingestion.
    """
    try:
        from constitutionlogic.constitution_digester import ConstitutionDigester, EMBEDDABLE_CHUNKS_JSON
        
        status_info = {
            'json_exists': False,
            'json_path': str(EMBEDDABLE_CHUNKS_JSON),
            'articles_extracted': 0,
            'chunks_in_json': 0,
            'chunks_in_db': 0,
            'db_accessible': True
        }
        
        # Check JSON file
        if EMBEDDABLE_CHUNKS_JSON.exists():
            status_info['json_exists'] = True
            
            try:
                with open(EMBEDDABLE_CHUNKS_JSON, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    metadata = data.get('metadata', {})
                    chunks = data.get('embeddable_chunks', [])
                    
                    status_info['articles_extracted'] = metadata.get('total_articles', 0)
                    status_info['chunks_in_json'] = len(chunks)
            except Exception as e:
                status_info['json_error'] = str(e)
        
        # Check ChromaDB
        try:
            digester = ConstitutionDigester()
            result = digester.collection.get(
                where={"document_type": "constitution"}
            )
            status_info['chunks_in_db'] = len(result['ids'])
        except Exception as e:
            status_info['db_accessible'] = False
            status_info['db_error'] = str(e)
        
        return jsonify(status_info), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Status check failed: {str(e)}',
            'status': 'error'
        }), 500


@app.route('/api/digest/constitution/reset', methods=['DELETE'])
@require_role("admin")
def reset_constitution():
    """
    Remove all constitution chunks from ChromaDB.
    ⚠️  WARNING: This is irreversible!
    """
    try:
        from constitutionlogic.constitution_digester import ConstitutionDigester
        
        print(f"\n{'='*80}")
        print(f"⚠️  RESETTING CONSTITUTION DATA")
        print(f"{'='*80}\n")
        
        digester = ConstitutionDigester()
        
        # Get all constitution chunk IDs
        result = digester.collection.get(
            where={"document_type": "constitution"}
        )
        
        chunk_ids = result['ids']
        
        if not chunk_ids:
            print("ℹ️  No constitution chunks found in database\n")
            return jsonify({
                'message': 'No constitution chunks to delete',
                'deleted': 0,
                'status': 'success'
            }), 200
        
        print(f"🗑️  Deleting {len(chunk_ids)} constitution chunks...")
        
        # Delete chunks
        digester.collection.delete(ids=chunk_ids)
        
        print(f"✅ Successfully deleted {len(chunk_ids)} chunks\n")
        print(f"{'='*80}\n")
        
        return jsonify({
            'message': f'Deleted {len(chunk_ids)} constitution chunks',
            'deleted': len(chunk_ids),
            'status': 'success'
        }), 200
        
    except Exception as e:
        print(f"\n🚨 RESET ERROR: {str(e)}\n")
        return jsonify({
            'error': f'Reset failed: {str(e)}',
            'status': 'error'
        }), 500        


@app.route('/api/case-pdf', methods=['GET'])
def serve_case_pdf():
    """
    Serve a case PDF by case_identifier.
    Query: ?case_id=<case_identifier>
    
    Resolves the PDF path from ChromaDB metadata and streams the file.
    Opens in browser (Content-Disposition: inline) so it displays in a new tab.
    """
    import chromadb
    import urllib.parse
    
    case_id = request.args.get('case_id')
    
    if not case_id:
        return jsonify({'error': 'Missing case_id parameter'}), 400
    
    # Decode URL-encoded case_id
    case_id = urllib.parse.unquote(case_id)
    
    try:
        # Connect to ChromaDB to get source_file_path
        if not CASELAW_DB_DIR.exists():
            return jsonify({'error': 'Database not found'}), 404

        client = chromadb.PersistentClient(path=str(CASELAW_DB_DIR))
        collection = client.get_collection(name=CASELAW_COLLECTION_NAME)
        
        # Get one chunk with this case_identifier
        results = collection.get(
            where={"case_identifier": case_id},
            limit=1,
            include=["metadatas"]
        )
        
        if not results['ids'] or not results['metadatas']:
            return jsonify({'error': f'Case not found: {case_id}'}), 404
        
        metadata = results['metadatas'][0]
        source_file_path = metadata.get('source_file_path', '')
        file_name = metadata.get('file_name', '')
        
        # Try to resolve PDF file
        pdf_path = None

        # Option 0: Check SQLite sidecar for an OCR copy (served_file_path)
        try:
            import sqlite3 as _sqlite3
            _db = Path(__file__).resolve().parent / "locations.db"
            if _db.exists() and source_file_path:
                _conn = _sqlite3.connect(str(_db))
                _row = _conn.execute(
                    "SELECT served_file_path FROM chunk_locations "
                    "WHERE source_file_path = ? LIMIT 1",
                    (source_file_path,),
                ).fetchone()
                _conn.close()
                if _row and _row[0] and _row[0] != source_file_path:
                    _served = Path(_row[0])
                    if _served.exists():
                        pdf_path = _served
        except Exception:
            pass  # sidecar not critical; fall through to normal resolution

        # Option 1: Use source_file_path if available
        if not pdf_path and source_file_path:
            candidate = Path(source_file_path)
            if candidate.exists() and candidate.is_file():
                pdf_path = candidate

        # Option 2: Fallback to RES_IPSA_LOQUITUR_DIR + file_name.pdf
        if not pdf_path and file_name:
            # Try with .pdf extension
            candidate = RES_IPSA_LOQUITUR_DIR / f"{file_name}.pdf"
            if candidate.exists():
                pdf_path = candidate
            else:
                # Try without adding .pdf (in case file_name already has it)
                candidate = RES_IPSA_LOQUITUR_DIR / file_name
                if candidate.exists():
                    pdf_path = candidate
        
        if not pdf_path:
            return jsonify({
                'error': f'PDF file not found for case: {case_id}',
                'tried_paths': [source_file_path, str(RES_IPSA_LOQUITUR_DIR / f"{file_name}.pdf")]
            }), 404
        
        # Security: Ensure resolved path is under an allowed root directory.
        # Allowed roots: RES_IPSA_LOQUITUR_DIR (legacy) and SCC_CASES_DIR (new cases).
        try:
            pdf_path_resolved = pdf_path.resolve()
            allowed_roots = [RES_IPSA_LOQUITUR_DIR.resolve(), SCC_CASES_DIR.resolve()]
            in_allowed = any(
                os.path.commonpath([pdf_path_resolved, root]) == str(root)
                for root in allowed_roots
            )
            if not in_allowed:
                return jsonify({'error': 'Access denied: path outside allowed directory'}), 403
        except Exception as e:
            return jsonify({'error': f'Path resolution error: {str(e)}'}), 500
        
        # Stream the PDF file
        # Use Content-Disposition: inline so browser opens it in a tab (not download)
        return send_file(
            pdf_path_resolved,
            mimetype='application/pdf',
            as_attachment=False,  # inline (open in browser)
            download_name=pdf_path_resolved.name
        )
        
    except Exception as e:
        print(f"Error serving PDF for case_id={case_id}: {e}")
        return jsonify({'error': f'Failed to serve PDF: {str(e)}'}), 500


@app.route('/api/chunks/<chunk_id>/location', methods=['GET'])
def chunk_location(chunk_id):
    """
    Return page + bounding-box location data for a chunk.
    Source: backend/locations.db (written by build_case_chunks.py / constitutionbreaker.py).
    """
    import sqlite3
    db_path = Path(__file__).resolve().parent / "locations.db"
    if not db_path.exists():
        return jsonify({"error": "Location database not available — run build_case_chunks.py first"}), 404
    chunk_id = urllib_parse.unquote(chunk_id)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM chunk_locations WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": f"No location data for chunk: {chunk_id}"}), 404
    return jsonify({
        "chunk_id": chunk_id,
        "collection": row["collection"],
        "source_file_path": row["source_file_path"],
        "served_file_path": row["served_file_path"],
        "page_start": row["page_start"],
        "page_end": row["page_end"],
        "boxes": json.loads(row["boxes_json"] or "[]"),
        "page_dims": json.loads(row["page_dims_json"] or "{}"),
        "label": row["label"],
        "ocr": bool(row["ocr"]),
    }), 200


@app.route('/api/documents/<path:doc_key>/chunks', methods=['GET'])
def document_chunks(doc_key):
    """
    List all chunks recorded in locations.db whose source path contains doc_key.
    Useful for loading all page positions for a given case file.
    """
    import sqlite3
    db_path = Path(__file__).resolve().parent / "locations.db"
    if not db_path.exists():
        return jsonify({"error": "Location database not available"}), 404
    doc_key = urllib_parse.unquote(doc_key)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT chunk_id, collection, page_start, page_end, label, ocr "
        "FROM chunk_locations WHERE source_file_path LIKE ?",
        (f"%{doc_key}%",),
    ).fetchall()
    conn.close()
    return jsonify({
        "doc_key": doc_key,
        "chunks": [dict(r) for r in rows],
    }), 200


@app.route('/api/rule-pdf', methods=['GET'])
def serve_rule_pdf():
    """
    Serve statutory document PDFs (Constitution or Evidence Act).
    Query: ?doc_type=constitution&source=TheConstitutionOfKenya.pdf
    
    Opens in browser (Content-Disposition: inline) so it displays in a new tab.
    """
    import urllib.parse
    
    doc_type = request.args.get('doc_type')
    source = request.args.get('source')
    
    if not doc_type or not source:
        return jsonify({'error': 'Missing doc_type or source parameter'}), 400
    
    # Decode URL-encoded source
    source = urllib.parse.unquote(source)
    
    try:
        # Rules PDFs are in MasterRules directory
        rules_dir = Path(__file__).resolve().parent / "MasterRules"
        
        # Map to actual PDF files
        pdf_path = rules_dir / source
        
        if not pdf_path.exists():
            return jsonify({
                'error': f'PDF file not found: {source}',
                'tried_path': str(pdf_path)
            }), 404
        
        # Security: Ensure resolved path is under rules_dir
        try:
            pdf_path_resolved = pdf_path.resolve()
            allowed_root = rules_dir.resolve()
            
            if not str(pdf_path_resolved).startswith(str(allowed_root)):
                return jsonify({'error': 'Access denied: path outside allowed directory'}), 403
        except Exception as e:
            return jsonify({'error': f'Path resolution error: {str(e)}'}), 500
        
        # Stream the PDF file
        return send_file(
            pdf_path_resolved,
            mimetype='application/pdf',
            as_attachment=False,  # inline (open in browser)
            download_name=pdf_path_resolved.name
        )
        
    except Exception as e:
        print(f"Error serving rule PDF doc_type={doc_type}, source={source}: {e}")
        return jsonify({'error': f'Failed to serve PDF: {str(e)}'}), 500


@app.route('/api/qretrieve', methods=['POST'])
@require_role("advocate", "admin")
def QRetriever():
        """
        Executes the legal query classification + retrieval pipeline.
        Supports parallel dual retrieval: case law + statutory rules.
        """

        data = request.get_json(silent=True)

        if not data or "query" not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'query' in request body"
            }), 400

        user_query = data["query"]

        # Step 1: Run Qclassifier (includes case law retrieval)
        result = Qclassifier(user_query)
        
        if not result.get("success"):
            status_code = 500
            return jsonify(result), status_code
        
        # Step 2: Check if rules retrieval is needed (mode-aware routing)
        classification = result.get("classification", {})
        retrieval_mode = classification.get("retrieval_mode", "")
        target_components = classification.get("target_components", [])
        
        # Determine if statutory retrieval should be triggered
        should_fetch_statutory = (
            retrieval_mode in ["DIRECT_STATUTORY_LOOKUP", "HYBRID_RETRIEVAL"] or
            "RULES" in target_components or
            "STATUTORY_TEXT" in target_components
        )
        
        if should_fetch_statutory:
            mode_label = retrieval_mode if retrieval_mode else "component-based"
            print(f"\n[INFO] Statutory retrieval triggered (mode: {mode_label})")
            
            try:
                # Call rules retriever with classification data
                rules_result = retrieve_relevant_rules(classification, verbose=False)
                
                # Add rules to the retrieval section
                if "retrieval" not in result:
                    result["retrieval"] = {}
                
                result["retrieval"]["rules"] = rules_result.get("results", [])
                result["retrieval"]["rules_count"] = rules_result.get("count", 0)
                result["retrieval"]["rules_query_summary"] = rules_result.get("query_summary", {})
                result["retrieval"]["retrieval_mode"] = retrieval_mode  # Pass mode to frontend
                
                print(f"[INFO] Retrieved {result['retrieval']['rules_count']} statutory provisions")
                
            except Exception as e:
                print(f"[WARNING] Failed to retrieve statutory rules: {e}")
                import traceback
                traceback.print_exc()
                # Don't fail the whole request, just log the error
                result["retrieval"]["rules"] = []
                result["retrieval"]["rules_count"] = 0
                result["retrieval"]["rules_error"] = str(e)
        else:
            print(f"[INFO] Statutory retrieval not needed (mode: {retrieval_mode or 'none'})")
            if "retrieval" not in result:
                result["retrieval"] = {}
            result["retrieval"]["rules"] = []
            result["retrieval"]["rules_count"] = 0

        status_code = 200 if result.get("success") else 500
        return jsonify(result), status_code


@app.route('/api/generate-legal-answer', methods=['POST'])
@require_role("advocate", "admin")
def generate_legal_answer():
    """
    Grounded synthesis: coverage check → structured JSON synthesis → citation verification.

    Request body:
      query          str   user question
      retrieval      dict  output from /api/qretrieve
      retrieval_mode str
      classification dict  classifier output (for coverage check)
    """
    from coverage import check_coverage
    from citation_verifier import verify_and_clean

    try:
        data = request.get_json()
        user_query     = data.get("query", "")
        retrieval_data = data.get("retrieval", {})
        classification = data.get("classification", {})

        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        statutory_provisions = retrieval_data.get("rules", [])
        case_law_results     = retrieval_data.get("results", [])

        print(f"\n[INFO] generate-legal-answer: {user_query[:60]}…")
        print(f"[INFO] Sources: {len(statutory_provisions)} statutes, {len(case_law_results)} cases")

        # ── 1. Coverage check ────────────────────────────────────────────────
        coverage = check_coverage(classification)
        print(f"[INFO] Coverage: in={coverage['in_corpus']}, out={coverage['not_in_corpus']}")

        # ── 2. Build labelled source list ────────────────────────────────────
        sources = []  # {"id", "type", "label", "data"}
        source_blocks: list[str] = []
        # alias_map: chunk-level alias (e.g. "C1.2") → chunk_id string
        alias_map: dict = {}

        for i, rule in enumerate(statutory_provisions[:5], 1):
            sid   = f"S{i}"
            label = rule.get("document_label", "Unknown")
            an    = rule.get("article_number", "")
            hdr   = rule.get("article_header", "")
            stat_chunks = rule.get("chunks", [])

            sources.append({"id": sid, "type": "statute", "label": f"{label} s.{an}", "data": rule})

            chunk_lines = []
            for j, sch in enumerate(stat_chunks, 1):
                chunk_alias = f"S{i}.{j}"
                sec_num = sch.get("section_num") or an
                sec_hdr = sch.get("section_heading") or hdr
                chunk_label = f"{label} s.{sec_num}"
                alias_map[chunk_alias] = {
                    "chunk_id": sch.get("chunk_id", ""),
                    "label": chunk_label,
                    "type": "statute",
                }
                chunk_lines.append(f"  [{chunk_alias}] s.{sec_num} {sec_hdr}\n  {sch.get('text','')[:600]}")

            # Fallback top-level alias for bare [S#] references
            top_cid = stat_chunks[0]["chunk_id"] if stat_chunks else rule.get("chunk_id", "")
            alias_map[sid] = {"chunk_id": top_cid, "label": f"{label} s.{an}", "type": "statute"}

            block_header = f"[{sid}] {label}, s.{an} — {hdr}"
            source_blocks.append(block_header + "\n" + "\n".join(chunk_lines))

        for i, case in enumerate(case_law_results[:3], 1):
            cid_base = f"C{i}"
            cn   = case.get("case_name", "Unknown")
            meta = case.get("metadata", {})
            year = meta.get("year", "")
            court= meta.get("court_level", "")
            case_chunks = case.get("chunks", [])[:3]
            chunk_lines = []
            for j, ch in enumerate(case_chunks, 1):
                chunk_alias = f"C{i}.{j}"
                chunk_cid = ch.get("chunk_id", ch.get("id", ""))
                alias_map[chunk_alias] = {
                    "chunk_id": chunk_cid,
                    "label": f"{cn} ({year}) chunk {j}",
                    "type": "case",
                }
                chunk_lines.append(f"  [{chunk_alias}] {ch.get('text','')[:500]}")
            sources.append({"id": cid_base, "type": "case", "label": f"{cn} ({year}, {court})", "data": case})
            source_blocks.append(
                f"[{cid_base}] {cn} ({year}, {court})\n" + "\n".join(chunk_lines)
            )

        if not source_blocks:
            return jsonify({"success": False, "error": "No relevant legal information retrieved"}), 404

        # ── 3. System prompt — grounding rules only, NO legal content ────────
        SYSTEM = (
            "You are a Kenyan legal research assistant. "
            "Output ONLY a single valid JSON object. No text before or after it.\n\n"
            "ABSOLUTE RULES:\n"
            "1. Every legal statement MUST cite using exact aliases from the sources.\n"
            "   Statute chunk aliases: [S1.1], [S1.2] etc. (use the most specific chunk alias).\n"
            "   Bare [S#] may be used to refer to a whole statute article if you didn't use a specific chunk.\n"
            "   Case chunks use [C#.#] (e.g. [C1.2]).\n"
            "2. Never state a section number, article number, rule number, case name, holding, "
            "   or procedure that does not appear verbatim or near-verbatim in the source text supplied.\n"
            "3. Never say a source is wrong, outdated, or misapplied. If the user's premise "
            "   conflicts with a source, report what the source says.\n"
            "4. If a provision the user names is NOT in the provided sources, state that in "
            "   premise_check and gaps. Do not describe its content from memory.\n"
            "5. For cases, report only what the judgment text explicitly says — parties, court, "
            "   outcome, quoted reasoning. Do not add, correct, or supplement.\n"
            "6. Use ONLY sources you mark relevant=true. After marking a source irrelevant, "
            "   do not reference it anywhere else.\n"
            "7. A short, accurate answer is better than a complete-looking one. "
            "   If sources are insufficient, say so and stop.\n\n"
            "OUTPUT SCHEMA (all fields required, empty string if not applicable):\n"
            "{\n"
            '  "source_review": [\n'
            '    {"id":"S1","relevant":true,"reason":"one sentence","key_points":["verbatim text"]}\n'
            "  ],\n"
            '  "premise_check": "Does the question rely on provisions not in the sources? List them.",\n'
            '  "short_answer": "2-3 sentences citing [S#]/[C#]. If insufficient: say so.",\n'
            '  "applicable_law": "Verbatim or near-verbatim text from relevant [S#] sources.",\n'
            '  "case_law_discussion": "Holdings as stated in [C#] sources — parties, court, outcome, quoted reasoning.",\n'
            '  "application": "How [S#][C#] provisions apply. No claims beyond source text.",\n'
            '  "jurisdiction_and_procedure": "From [S#][C#] only. Empty string if not in sources.",\n'
            '  "gaps": "What the question needed that sources do not cover.",\n'
            '  "follow_up_questions": ["up to 3 questions the corpus could answer"],\n'
            '  "citations_used": ["S1.1 — Document s.N chunk", "C1.2 — Case name chunk"]\n'
            "}"
        )

        # ── 4. User prompt ────────────────────────────────────────────────────
        coverage_notice = ""
        if coverage["not_in_corpus"]:
            acts = "; ".join(coverage["not_in_corpus"])
            coverage_notice = (
                f"COVERAGE NOTICE: The following Acts named in the question are NOT in the "
                f"knowledge base — do not describe their content: {acts}.\n\n"
            )

        USER = (
            coverage_notice
            + "SOURCES:\n"
            + "\n\n".join(source_blocks)
            + f'\n\nUSER QUESTION: "{user_query}"\n\nOutput the JSON object now.'
        )

        # ── 5. Groq call (temperature 0, with key rotation) ──────────────────
        _groq_keys = [v for k, v in sorted(os.environ.items())
                      if k == "GROQ_API_KEY" or k.startswith("GROQ_API_KEY_")]
        if not _groq_keys:
            return jsonify({"success": False, "error": "No GROQ_API_KEY configured"}), 500
        _key_idx = [0]

        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]

        def _groq(msgs, max_tokens=2500) -> str:
            GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
            for attempt in range(len(_groq_keys) + 1):
                key = _groq_keys[_key_idx[0]]
                try:
                    resp = requests.post(
                        GROQ_URL,
                        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                        json={"model": "qwen/qwen3.8-27b", "messages": msgs,
                              "temperature": 0, "max_tokens": max_tokens},
                        timeout=60,
                    )
                    if resp.status_code == 429:
                        next_idx = (_key_idx[0] + 1) % len(_groq_keys)
                        if next_idx != _key_idx[0]:
                            print(f"[WARN] synthesis 429 on key {_key_idx[0]}, rotating to {next_idx}")
                            _key_idx[0] = next_idx
                            continue
                        print("[WARN] All synthesis keys rate-limited, waiting 30s...")
                        time.sleep(30)
                        continue
                    resp.raise_for_status()
                    raw = resp.json()["choices"][0]["message"]["content"]
                    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                except requests.HTTPError:
                    raise
            raise RuntimeError("All Groq keys exhausted for synthesis")

        raw = _groq(messages)

        # ── 6. Parse JSON ─────────────────────────────────────────────────────
        def _parse_json(text: str) -> dict:
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
            text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)
            start = text.find("{")
            if start == -1:
                raise ValueError("No JSON found")
            depth = 0
            for i, ch in enumerate(text[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return json.loads(text[start: i + 1])
            raise ValueError("Unmatched braces")

        try:
            parsed = _parse_json(raw)
        except Exception as parse_err:
            print(f"[WARN] JSON parse failed ({parse_err}); returning raw text")
            parsed = {"short_answer": raw, "source_review": []}

        # ── 7. Filter irrelevant sources ──────────────────────────────────────
        relevant_ids = {
            sr["id"]
            for sr in parsed.get("source_review", [])
            if sr.get("relevant") is True
        }
        # If model didn't mark anything relevant, treat all as relevant
        if not relevant_ids:
            relevant_ids = {s["id"] for s in sources}

        relevant_source_data = [s["data"] for s in sources if s["id"] in relevant_ids]

        # ── 8. Citation verification ──────────────────────────────────────────
        analysis_fields = " ".join(filter(None, [
            parsed.get("short_answer", ""),
            parsed.get("applicable_law", ""),
            parsed.get("case_law_discussion", ""),
            parsed.get("application", ""),
            parsed.get("jurisdiction_and_procedure", ""),
        ]))

        def _regen(unsupported_list: list) -> str:
            regen_msgs = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    "The following citations are not present in the provided sources and must be "
                    f"removed or replaced with 'not in the sources': {', '.join(unsupported_list)}. "
                    "Output the corrected JSON object."
                )},
            ]
            return _groq(regen_msgs)

        verified = verify_and_clean(analysis_fields, relevant_source_data, user_query, _regen)

        removed = verified.get("unverified_claims_removed", [])
        if removed:
            print(f"[WARN] Removed {len(removed)} unverified claim(s): {removed}")

        # ── 9. Build answer string ────────────────────────────────────────────
        def _build_answer(p: dict) -> str:
            parts: list[str] = []

            for key, heading in [
                ("short_answer",              None),
                ("applicable_law",            "#### Applicable Law"),
                ("case_law_discussion",       "#### Case Law"),
                ("application",               "#### Application"),
                ("jurisdiction_and_procedure","#### Jurisdiction & Procedure"),
                ("gaps",                      "#### Gaps in Available Sources"),
            ]:
                val = (p.get(key) or "").strip()
                skip = {"", "n/a", "none", "not applicable", "no gaps identified", "no gaps"}
                if val.lower() not in skip:
                    parts.append(f"{heading}\n{val}" if heading else val)

            fqs = p.get("follow_up_questions") or []
            if fqs:
                parts.append("#### Follow-up Questions\n" + "\n".join(f"- {q}" for q in fqs[:3]))

            cu = p.get("citations_used") or []
            if cu:
                parts.append("#### Sources Referenced\n" + "\n".join(f"- {c}" for c in cu))

            return "\n\n".join(parts)

        answer = _build_answer(parsed)

        # Determine which chunk aliases appear in the answer text
        _alias_pat = re.compile(r"\[([CS]\d+(?:\.\d+)?)\]")
        _used_aliases = set(_alias_pat.findall(answer))
        # Also check all parsed text fields
        for _field in ("applicable_law", "case_law_discussion", "application",
                        "short_answer", "jurisdiction_and_procedure"):
            _used_aliases.update(_alias_pat.findall(parsed.get(_field) or ""))

        response_sources = []
        for alias, info in alias_map.items():
            base = alias.split(".")[0]  # e.g. "C1" or "S1"
            is_chunk_alias = "." in alias  # S1.1 / C1.2 — chunk-level
            cited_directly = alias in _used_aliases
            cited_via_parent = base in _used_aliases  # bare [C1]/[S1] in answer

            used = cited_directly or cited_via_parent

            if cited_directly:
                precision = "chunk"
            elif cited_via_parent and is_chunk_alias:
                precision = "document"
            elif cited_directly and not is_chunk_alias:
                precision = "document"
            else:
                precision = "chunk" if is_chunk_alias else "document"

            response_sources.append({
                "alias": alias,
                "chunk_id": info["chunk_id"],
                "label": info["label"],
                "type": info["type"],
                "used": used,
                "precision": precision,
            })

        print(f"[SUCCESS] Answer built ({len(answer)} chars), removed={len(removed)}")

        return jsonify({
            "success": True,
            "answer": answer,
            "coverage": coverage,
            "source_review": parsed.get("source_review", []),
            "premise_check": (parsed.get("premise_check") or "").strip(),
            "unverified_claims_removed": removed,
            "sources": response_sources,
            "alias_map": {k: v["chunk_id"] for k, v in alias_map.items()},
            "sources_used": {
                "statutory_provisions": len(statutory_provisions),
                "case_law": len(case_law_results),
            },
        }), 200

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] LLM API call failed: {e}")
        return jsonify({"success": False, "error": f"LLM call failed: {e}"}), 500

    except Exception as e:
        import traceback
        print(f"[ERROR] generate-legal-answer: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Case Laws repository endpoints
# ---------------------------------------------------------------------------

ALLOWED_CASE_EXTS = {'pdf', 'docx'}


@app.route('/api/case-laws/upload', methods=['POST'])
@require_role("advocate", "admin")
def upload_case_law():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in ALLOWED_CASE_EXTS:
        return jsonify({'error': 'Only PDF and DOCX files are allowed'}), 400

    SCC_CASES_DIR.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(file.filename)

    file_data = file.read()
    file_size = len(file_data)

    # Persist locally for RAG pipeline
    with open(SCC_CASES_DIR / filename, 'wb') as fh:
        fh.write(file_data)

    # Write metadata to Firestore CaseLaws collection
    title = filename.rsplit('.', 1)[0] if '.' in filename else filename
    local_path = str(SCC_CASES_DIR / filename)
    db_client = fb_firestore.client()
    _, doc_ref = db_client.collection('CaseLaws').add({
        'title': title,
        'fileName': filename,
        'fileType': ext,
        'localPath': local_path,
        'uploadedBy': g.profile.get('username', 'unknown'),
        'uploadedByUid': g.uid,
        'uploadedAt': fb_firestore.SERVER_TIMESTAMP,
        'fileSizeBytes': file_size,
    })

    return jsonify({
        'id': doc_ref.id,
        'title': title,
        'fileName': filename,
        'localPath': local_path,
    }), 200


@app.route('/api/case-laws/file/<path:filename>', methods=['GET'])
def serve_case_law_file(filename):
    """Serve a case law file from SCCCases/ (local, no auth — internal tool only)."""
    # Resolve and guard against directory traversal without mangling the original name
    resolved = (SCC_CASES_DIR / filename).resolve()
    if not str(resolved).startswith(str(SCC_CASES_DIR.resolve())):
        return jsonify({'error': 'Invalid path'}), 400
    if not resolved.exists():
        return jsonify({'error': 'File not found'}), 404

    ext = resolved.suffix.lower().lstrip('.')
    if ext not in ALLOWED_CASE_EXTS:
        return jsonify({'error': 'File type not permitted'}), 400

    mime = (
        'application/pdf' if ext == 'pdf'
        else 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    )
    return send_from_directory(str(SCC_CASES_DIR), resolved.name, mimetype=mime, as_attachment=False)


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')

