#!/usr/bin/env python3
"""
Quick demonstration of the enhanced PDF loading capabilities
Shows how to load both .txt and .pdf files with the RAG system
"""

import os
from pixe import load_document, chunk_text

def demo_pdf_loading():
    """Demonstrate loading and processing PDF files"""
    
    print("\n" + "="*70)
    print("🎯 ENHANCED DOCUMENT LOADER DEMO")
    print("="*70)
    
    # Check available files
    pdf_file = "Bobs_superheroes.pdf"
    txt_file = "tmnt.txt"
    
    # Demo 1: Load and chunk a PDF
    if os.path.exists(pdf_file):
        print("\n📄 DEMO 1: Loading PDF File")
        print("-" * 70)
        
        documents = load_document(pdf_file)
        
        if documents:
            doc = documents[0]
            print(f"\n✅ Successfully loaded: {doc.metadata['file_name']}")
            print(f"   File type: {doc.metadata['file_type']}")
            print(f"   Original size: {doc.metadata['original_size']:,} characters")
            print(f"   After cleaning: {doc.metadata['file_size']:,} characters")
            print(f"   Noise removed: {((doc.metadata['original_size'] - doc.metadata['file_size']) / doc.metadata['original_size'] * 100):.1f}%")
            
            # Chunk the document
            print("\n📊 Chunking the document...")
            chunks, chunk_docs = chunk_text(documents)
            
            print(f"✅ Created {len(chunks)} chunks")
            print(f"\n📝 Sample chunk (first 300 chars):")
            print("-" * 70)
            print(chunks[0][:300] + "...")
            print("-" * 70)
    else:
        print(f"⚠️  PDF file not found: {pdf_file}")
    
    # Demo 2: Load and chunk a text file for comparison
    if os.path.exists(txt_file):
        print("\n📄 DEMO 2: Loading Text File (for comparison)")
        print("-" * 70)
        
        documents = load_document(txt_file)
        
        if documents:
            doc = documents[0]
            print(f"\n✅ Successfully loaded: {doc.metadata['file_name']}")
            print(f"   File type: {doc.metadata['file_type']}")
            print(f"   File size: {doc.metadata['file_size']:,} characters")
            
            # Chunk the document
            chunks, chunk_docs = chunk_text(documents)
            print(f"✅ Created {len(chunks)} chunks")
    else:
        print(f"⚠️  Text file not found: {txt_file}")
    
    # Summary
    print("\n" + "="*70)
    print("📚 SUMMARY")
    print("="*70)
    print("\nThe enhanced document loader supports:")
    print("  ✅ .txt files - Plain text with UTF-8 encoding")
    print("  ✅ .pdf files - Automatic text extraction from all pages")
    print("  ✅ Text cleaning - Removes noise and formatting artifacts")
    print("  ✅ Smart chunking - Preserves context with overlap")
    print("  ✅ Metadata tracking - Full citation support")
    print("\n💡 Usage:")
    print("  from pixe import load_document, chunk_text")
    print("  documents = load_document('yourfile.pdf')")
    print("  chunks, chunk_docs = chunk_text(documents)")
    print("\n📖 See DOCUMENT_LOADING_GUIDE.md for detailed documentation")
    print("="*70 + "\n")

if __name__ == "__main__":
    demo_pdf_loading()

