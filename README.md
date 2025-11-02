# Retrieval-Augmented Generation (RAG) AI

A comprehensive RAG system that uses local embeddings and remote LLMs via Hugging Face API for intelligent document question-answering.

## Features

- **Enhanced Semantic Search**: Uses `intfloat/e5-base-v2` for better understanding (768 dimensions vs 384)
- **Hybrid Search**: Combines semantic similarity (70%) + keyword matching (30%)
- **Query Expansion**: Automatically expands queries with synonyms and related terms
- **Two-Stage Retrieval**: Broad retrieval followed by precise reranking using cross-encoder
- **Smart Chunking**: Improved text chunking with better overlap and metadata preservation
- **Citation Tracking**: Tracks sources for all retrieved information
- **Interactive Mode**: Run Q&A sessions on your documents

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Retreival-Augmented-Generation-AI.git
cd Retreival-Augmented-Generation-AI
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages**:
- `python-dotenv` - Environment variable management
- `sentence-transformers` - Embedding models
- `huggingface_hub` - Hugging Face API client
- `langchain` - Document processing
- `scikit-learn` - Similarity metrics
- `numpy` - Numerical operations

### 4. Configure Hugging Face Token

1. Get your Hugging Face token from: https://huggingface.co/settings/tokens
2. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
3. Edit `.env` and replace `your_huggingface_token_here` with your actual token:
```env
HF_API_TOKEN=hf_your_actual_token_here
```

**Important**: Never commit your `.env` file! It's already in `.gitignore`.

## Usage

### Interactive Mode

Run the interactive RAG session:

```bash
python pixe.py
```

Or with the simpler mistral.py:

```bash
python mistral.py
```

The system will:
1. Load and chunk your document
2. Create embeddings
3. Wait for your questions
4. Retrieve relevant chunks and generate answers

### Programmatic Usage

```python
from pixe import run_rag_pipeline

# Single query
answer = run_rag_pipeline('tmnt.txt', 'Who are the Teenage Mutant Ninja Turtles?')
print(answer)
```

### Custom Configuration

Edit the configuration section in `pixe.py`:

```python
# LLM Model
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Embedding Model
EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"

# Chunking Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300

# Retrieval Settings
TOP_K = 3  # Number of final chunks to retrieve
INITIAL_RETRIEVAL_K = 10  # Initial candidates for reranking

# Feature Toggles
USE_RERANKING = True
USE_QUERY_EXPANSION = True
USE_HYBRID_SEARCH = True
```

## Supported Document Formats

Currently supports:
- Plain text files (`.txt`)
- Can be extended for PDFs, markdown, etc.

## Models Used

### LLM (Remote via Hugging Face)
- **Mistral-7B-Instruct-v0.2**: Used for query expansion and answer generation
- Accessed via Hugging Face Inference API
- No local GPU required!

### Embeddings (Local)
- **intfloat/e5-base-v2**: 768-dimensional embeddings
- Runs on CPU or GPU (automatic detection)
- Alternative options: `all-mpnet-base-v2`, `BAAI/bge-base-en-v1.5`

### Reranking (Local)
- **cross-encoder/ms-marco-MiniLM-L-6-v2**: Two-stage retrieval refinement
- Lightweight cross-encoder model
- Improves precision of retrieved chunks

## Performance

### On Normal Laptops (Intel 11th Gen, 16GB RAM)
- ✅ Works excellently via remote Hugging Face API
- ✅ Local embeddings and reranking are CPU-friendly
- ✅ No GPU required for LLM inference
- ⚠️ First run downloads models (~500MB), subsequent runs are fast

### On Company Servers
- ✅ Should work perfectly
- ✅ Remote API handles heavy LLM workload
- ✅ Can scale to multiple users

### If You Download Models Locally
- **CPU-only**: 3-5 tokens/sec for Mistral-7B (not recommended for production)
- **GPU with 8GB VRAM**: Smooth operation with 4-bit quantization
- **GPU with 16GB+ VRAM**: Excellent performance with FP16 or 8-bit
- Consider using Ollama or llama.cpp for easier local deployment

## File Structure

```
Retreival-Augmented-Generation-AI/
├── pixe.py              # Enhanced RAG system (full features)
├── mistral.py           # Simplified RAG system
├── .env                 # Your HF token (not in git)
├── .env.example         # Template for .env
├── .gitignore           # Git ignore rules
├── requirements.txt     # Python dependencies
├── tmnt.txt             # Sample document
├── essay.txt            # Sample document
├── Bobs_superheroes.pdf # Sample PDF
├── venv/                # Virtual environment
└── README.md            # This file
```

## Troubleshooting

### "HF_API_TOKEN not found"
- Make sure `.env` file exists
- Check that your token is correctly set in `.env`
- Verify `python-dotenv` is installed

### Import Errors
- Activate your virtual environment
- Install dependencies: `pip install -r requirements.txt`

### Slow Performance
- First run downloads models (one-time)
- For local LLM inference, ensure sufficient RAM/VRAM
- Consider using smaller models for faster inference

### Out of Memory
- Reduce `CHUNK_SIZE` or `INITIAL_RETRIEVAL_K`
- Process smaller documents
- Use smaller embedding model

## Contributing

Pull requests welcome! Please ensure:
1. Your code follows PEP 8 style guide
2. Add docstrings to new functions
3. Test your changes before submitting
4. Keep `.env` out of commits!

## License

MIT License - feel free to use this for your projects!

## Acknowledgments

- Hugging Face for models and API
- Sentence Transformers for embeddings
- LangChain for document processing

