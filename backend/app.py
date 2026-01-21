from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import time
import os
import sys
import io
import json
from pathlib import Path
from werkzeug.utils import secure_filename
from digester import digest_repo
from pixe import run_rag_pipeline
from CaseLaw.firac import run_firac, run_firac_from_file
from CaseLaw.ingester import ingest_firac_data
from Qclassifier import Qclassifier
from constitutionlogic.constitutionbreaker import extract_constitution_articles, OUTPUT_DIR as CONST_OUTPUT_DIR, flatten_articles_to_embeddable_chunks, CHUNK_SIZE, CHUNK_OVERLAP
from constitutionlogic.constitution_digester import ConstitutionDigester, EMBEDDABLE_CHUNKS_JSON
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Path to the Repo directory
REPO_DIR = Path(__file__).resolve().parent / "Repo"
RES_IPSA_LOQUITUR_DIR = REPO_DIR / "Res_ipsa_loquitur"

@app.route('/api/chat', methods=['POST'])
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
def get_files():
    """
    Return list of files in the Repo directory.
    """
    if not REPO_DIR.exists():
        return jsonify({'files': []})
    
    files = [f.name for f in REPO_DIR.iterdir() if f.is_file()]
    return jsonify({'files': files})

@app.route('/api/upload', methods=['POST'])
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


@app.route('/api/qretrieve', methods=['POST'])
def QRetriever():
    """
    Executes the legal query classification + retrieval pipeline.
    """

    data = request.get_json(silent=True)

    if not data or "query" not in data:
        return jsonify({
            "success": False,
            "error": "Missing 'query' in request body"
        }), 400

    user_query = data["query"]

    result = Qclassifier(user_query)

    status_code = 200 if result.get("success") else 500
    return jsonify(result), status_code
      
if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')

