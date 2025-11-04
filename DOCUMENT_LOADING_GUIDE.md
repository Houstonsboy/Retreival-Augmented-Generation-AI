# Enhanced Document Loading Guide

## Overview

The RAG system now supports loading, extracting, cleaning, and chunking content from multiple file formats:
- ✅ **Text files** (.txt)
- ✅ **PDF files** (.pdf)

## Features

### 1. **Multi-Format Support**
- Automatically detects file type based on extension
- Handles `.txt` files with UTF-8 encoding
- Extracts text from `.pdf` files page by page

### 2. **Intelligent Text Extraction**
- **PDF Extraction**: Uses `pypdf` to extract text from each page
- Progress tracking showing extraction per page
- Handles multi-page PDFs efficiently

### 3. **Text Cleaning**
- Removes navigation elements (Search, Sign In, Menu, etc.)
- Normalizes whitespace
- Removes page numbers and decorative characters
- Preserves semantic content
- Reports cleaning statistics (% of noise removed)

### 4. **Intelligent Chunking**
- Splits documents into manageable chunks (1000 chars with 300 char overlap)
- Respects semantic boundaries (paragraphs, sentences)
- Preserves metadata for citation tracking
- Tracks chunk positions and sources

### 5. **Metadata Preservation**
- Tracks source file, file type, and file name
- Records original and cleaned file sizes
- Maintains chunk indices and start positions
- Enables precise citation tracking

## Installation

Install the required PDF library:

```bash
pip install pypdf>=3.17.0
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Usage Examples

### Example 1: Load a Text File

```python
from pixe import load_document, chunk_text

# Load a .txt file
documents = load_document("tmnt.txt")

# Chunk the document
chunks, chunk_documents = chunk_text(documents)

print(f"Loaded {len(chunks)} chunks from text file")
```

### Example 2: Load a PDF File

```python
from pixe import load_document, chunk_text

# Load a .pdf file
documents = load_document("Bobs_superheroes.pdf")

# Chunk the document
chunks, chunk_documents = chunk_text(documents)

print(f"Loaded {len(chunks)} chunks from PDF")
print(f"Metadata: {chunk_documents[0].metadata}")
```

### Example 3: Full RAG Pipeline with PDF

```python
from pixe import run_rag_pipeline

# Run the full RAG pipeline on a PDF
answer = run_rag_pipeline(
    file_path="Bobs_superheroes.pdf",
    query="What is the main plot of the story?"
)

print(answer)
```

### Example 4: Interactive Mode with PDF

```python
from pixe import interactive_mode

# Start interactive Q&A session with a PDF
interactive_mode("Bobs_superheroes.pdf")
```

## What Happens Under the Hood

### Loading Process Flow

```
1. File Detection
   ├── Check if file exists
   ├── Detect file extension (.txt or .pdf)
   └── Route to appropriate loader

2. Text Extraction
   ├── For .txt: Read with UTF-8 encoding
   └── For .pdf: Extract text page by page using pypdf

3. Text Cleaning
   ├── Remove navigation/UI elements
   ├── Normalize whitespace
   ├── Remove noise (page numbers, etc.)
   └── Report cleaning statistics

4. Document Creation
   ├── Create LangChain Document object
   ├── Attach metadata (source, type, size, etc.)
   └── Return document list

5. Chunking
   ├── Split into manageable chunks (1000 chars)
   ├── Use overlapping chunks (300 chars overlap)
   ├── Respect semantic boundaries
   ├── Add chunk-specific metadata
   └── Return chunks and chunk documents
```

## Test Results

### Text File (.txt)
- ✅ Successfully loaded `tmnt.txt`
- 6,620 characters extracted
- 0.1% removed during cleaning
- Created 12 chunks

### PDF File (.pdf)
- ✅ Successfully loaded `Bobs_superheroes.pdf`
- 49 pages processed
- 138,666 characters extracted
- 21.4% removed during cleaning (navigation/noise)
- Created 157 chunks with full metadata

## Metadata Structure

Each document chunk includes comprehensive metadata:

```python
{
    'source': 'Bobs_superheroes.pdf',           # Original file path
    'file_type': '.pdf',                        # File extension
    'file_name': 'Bobs_superheroes.pdf',       # File name only
    'file_size': 109004,                        # Cleaned size (chars)
    'original_size': 138666,                    # Raw extracted size
    'loaded_at': '2025-11-04 13:56:38',        # Load timestamp
    'cleaned': True,                            # Was text cleaned?
    'start_index': 410,                         # Chunk start position
    'chunk_id': 1,                              # Chunk identifier
    'chunk_index': 1,                           # Chunk position
    'total_chunks': 157,                        # Total chunks in document
    'chunk_size': 998                           # Chunk size (chars)
}
```

## Advanced Configuration

You can customize the chunking behavior by modifying constants in `pixe.py`:

```python
CHUNK_SIZE = 1000        # Size of each chunk in characters
CHUNK_OVERLAP = 300      # Overlap between chunks for context
```

## Error Handling

The system handles various error scenarios:

- **File not found**: Returns empty list with error message
- **Unsupported format**: Attempts to read as text file
- **Empty PDF**: Detects and reports no extractable text
- **Image-based PDFs**: Reports pages with no text content
- **Encoding errors**: Handles with appropriate error messages

## Benefits of Enhanced Loading

### 1. **Broader Document Support**
- Work with PDFs without manual conversion
- Process text files as before
- Extensible for future formats

### 2. **Better Semantic Understanding**
- Text cleaning removes noise
- Improves embedding quality
- Better search relevance

### 3. **Precise Citations**
- Track source file and chunk position
- Reference specific pages/sections
- Enable verifiable answers

### 4. **Robust Processing**
- Handles large documents efficiently
- Page-by-page processing for PDFs
- Progress tracking and error reporting

## Future Enhancements

Potential future additions:
- DOCX support (Microsoft Word)
- HTML/Markdown support
- Image-based PDF support (OCR)
- Excel/CSV support for tabular data
- Archive support (.zip containing documents)

## Troubleshooting

### Issue: No text extracted from PDF
**Cause**: PDF might be image-based or encrypted  
**Solution**: Use OCR tools to convert to searchable PDF

### Issue: Chunking creates too many/few chunks
**Cause**: Document size vs chunk size mismatch  
**Solution**: Adjust CHUNK_SIZE and CHUNK_OVERLAP

### Issue: Text cleaning removes too much content
**Cause**: Overly aggressive cleaning rules  
**Solution**: Modify navigation_words list in clean_text()

## Summary

The enhanced document loading system provides:
- 📄 Multi-format support (.txt, .pdf)
- 🧹 Intelligent text cleaning
- 📊 Comprehensive metadata tracking
- 🔍 Better semantic search through cleaned content
- 📚 Scalable chunking for large documents
- ✅ Production-ready error handling

Test it with your own documents and see the improved RAG performance!

