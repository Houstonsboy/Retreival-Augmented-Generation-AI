from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import time
import os
import sys
import io
from pathlib import Path
from werkzeug.utils import secure_filename
from digester import digest_repo
from pixe import run_rag_pipeline
from firac import run_firac
from ingester import ingest_firac_data

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Path to the Repo directory
REPO_DIR = Path(__file__).resolve().parent / "Repo"

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
    Trigger the digester to process files in the Repo directory.
    The digester will only process new or changed files (based on file hash).
    """
    try:
        # Capture stdout to get digest output
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            # Run the digester
            digest_repo(repo_dir=REPO_DIR)
            output = captured_output.getvalue()
            
            return jsonify({
                'message': 'Digest completed successfully',
                'output': output,
                'status': 'success'
            }), 200
        except FileNotFoundError as e:
            return jsonify({
                'error': f'Repository directory not found: {str(e)}',
                'status': 'error'
            }), 404
        except Exception as e:
            output = captured_output.getvalue()
            return jsonify({
                'error': f'Digest failed: {str(e)}',
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

@app.route('/api/firac', methods=['POST'])
def firac():
    """
    Execute firac.py to process the Wilson Wanjala PDF document.
    Returns the extracted and cleaned text content.
    """
    try:
        # Capture stdout to get processing output (logs/print statements)
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            # Run the FIRAC processor (combined call)
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
            return jsonify({
                'error': f'Wilson Wanjala PDF not found: {str(e)}',
                'content': 'Error: Wilson Wanjala PDF file not found in the repository.',
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
    Accepts FIRAC data from the frontend and stores each component as a separate chunk.
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No FIRAC data provided',
                'status': 'error'
            }), 400
        
        # Capture stdout to get ingestion output
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            # Reconstruct FIRAC data structure from frontend format
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
            
            # Get optional source file path if provided
            source_file_path = data.get('source_file_path')
            case_identifier = data.get('case_identifier')
            
            # Ingest the FIRAC data
            ingest_firac_data(
                firac_data=firac_data,
                case_identifier=case_identifier,
                source_file_path=source_file_path
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


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')

