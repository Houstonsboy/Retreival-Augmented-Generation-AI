# Quick Reference - Enhanced Document Loading

## 🚀 Installation

```bash
pip install pypdf>=3.17.0
```

## 📖 Basic Usage

### Load Any Document

```python
from pixe import load_document

# Works with both .txt and .pdf
docs = load_document("myfile.pdf")
docs = load_document("myfile.txt")
```

### Load and Chunk

```python
from pixe import load_document, chunk_text

docs = load_document("document.pdf")
chunks, chunk_docs = chunk_text(docs)

print(f"Created {len(chunks)} chunks")
```

### Full RAG Pipeline

```python
from pixe import run_rag_pipeline

answer = run_rag_pipeline(
    file_path="document.pdf",
    query="What is this about?"
)
print(answer)
```

### Interactive Q&A

```python
from pixe import interactive_mode

interactive_mode("document.pdf")
# Now ask questions interactively!
```

## 📊 What You Get

### For Text Files (.txt)
```python
documents = load_document("file.txt")
# ✅ UTF-8 text loaded
# ✅ Cleaned and normalized
# ✅ Ready for chunking
```

### For PDF Files (.pdf)
```python
documents = load_document("file.pdf")
# ✅ All pages extracted
# ✅ Text cleaned (navigation, noise removed)
# ✅ Metadata preserved (pages, positions)
# ✅ Ready for chunking
```

## 🔍 Inspect Metadata

```python
from pixe import load_document

docs = load_document("file.pdf")
meta = docs[0].metadata

print(f"File: {meta['file_name']}")
print(f"Type: {meta['file_type']}")
print(f"Size: {meta['file_size']} chars")
print(f"Original: {meta['original_size']} chars")
print(f"Cleaned: {meta['cleaned']}")
```

## 🎯 Command Line Usage

```bash
# Run demo
python demo_pdf_loading.py

# Run interactive mode (edit pixe.py to set file)
python pixe.py
```

## ⚙️ Configuration

Edit `pixe.py` to customize:

```python
# Chunking settings
CHUNK_SIZE = 1000        # Chunk size in characters
CHUNK_OVERLAP = 300      # Overlap for context

# Retrieval settings
TOP_K = 3                # Final chunks to retrieve
INITIAL_RETRIEVAL_K = 10 # Initial candidates

# Features
USE_RERANKING = True
USE_QUERY_EXPANSION = True
USE_HYBRID_SEARCH = True
```

## 📁 Supported Formats

| Format | Extension | Status |
|--------|-----------|--------|
| Text   | `.txt`    | ✅ Supported |
| PDF    | `.pdf`    | ✅ Supported |
| Word   | `.docx`   | 🔮 Future |
| Markdown | `.md`   | 🔮 Future |

## 🐛 Common Issues

### PDF has no text
**Problem:** Image-based PDF  
**Solution:** Use OCR tools to convert first

### Import error for pypdf
**Problem:** Library not installed  
**Solution:** `pip install pypdf>=3.17.0`

### Too many chunks created
**Problem:** Large document  
**Solution:** Increase `CHUNK_SIZE` in pixe.py

### Not enough chunks
**Problem:** Small document or large chunk size  
**Solution:** Decrease `CHUNK_SIZE` in pixe.py

## 📚 More Info

- **Full Guide**: See `DOCUMENT_LOADING_GUIDE.md`
- **Changes**: See `CHANGES_SUMMARY.md`
- **General**: See `README.md`

## 💡 Tips

1. **PDFs work best** when they have extractable text (not just images)
2. **Larger PDFs** may take a few seconds to extract
3. **Cleaning removes** ~0-25% of content (mostly noise)
4. **Chunks overlap** to preserve context across boundaries
5. **Metadata tracking** enables precise citations

## 🎉 That's It!

You're ready to process both text and PDF files with your enhanced RAG system!

```python
from pixe import run_rag_pipeline

# Just change the filename - everything else is automatic!
answer = run_rag_pipeline("yourfile.pdf", "Your question here")
print(answer)
```

