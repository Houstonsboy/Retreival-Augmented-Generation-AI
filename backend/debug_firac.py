"""
Debug script to investigate why FIRAC extraction returns empty components.
Tests a single PDF file and shows detailed output.
"""
import sys
from pathlib import Path
from firac import run_firac_from_file

# Get PDF file path from command line or use first file
if len(sys.argv) > 1:
    pdf_path = Path(sys.argv[1])
else:
    # Use first PDF from Res_ipsa_loquitur
    repo_dir = Path(__file__).resolve().parent / "Repo" / "Res_ipsa_loquitur"
    pdf_files = list(repo_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found!")
        sys.exit(1)
    pdf_path = pdf_files[0]

print("=" * 80)
print(f"DEBUGGING FIRAC EXTRACTION FOR: {pdf_path.name}")
print("=" * 80)

# Run FIRAC extraction
result = run_firac_from_file(pdf_path)

# Display results
print("\n" + "=" * 80)
print("EXTRACTION RESULTS")
print("=" * 80)

if result.get('error'):
    print(f"❌ ERROR: {result['error']}")
else:
    print("\n📊 Component Lengths:")
    print(f"   • Document: {len(result.get('document', ''))} chars")
    print(f"   • Metadata: {len(result.get('metadata', ''))} chars")
    print(f"   • Facts: {len(result.get('facts', ''))} chars")
    print(f"   • Issues: {len(result.get('issues', ''))} chars")
    print(f"   • Rules: {len(result.get('rules', ''))} chars")
    print(f"   • Application: {len(result.get('application', ''))} chars")
    print(f"   • Conclusion: {len(result.get('conclusion', ''))} chars")
    
    # Check which components are empty
    empty_components = []
    for component in ['facts', 'issues', 'rules', 'application', 'conclusion']:
        content = result.get(component, '').strip()
        if not content:
            empty_components.append(component)
    
    if empty_components:
        print(f"\n⚠️  EMPTY COMPONENTS: {', '.join(empty_components)}")
    else:
        print("\n✅ All FIRAC components have content")
    
    # Show full response preview
    print("\n" + "=" * 80)
    print("FULL API RESPONSE PREVIEW (first 2000 chars)")
    print("=" * 80)
    full_response = result.get('full_response', '')
    if full_response:
        print(full_response[:2000])
        if len(full_response) > 2000:
            print(f"\n... (truncated, total length: {len(full_response)} chars)")
    else:
        print("No full_response found!")
    
    # Check for section headers in response
    print("\n" + "=" * 80)
    print("SECTION HEADER DETECTION")
    print("=" * 80)
    section_patterns = [
        ("SECTION 0: METADATA", "SECTION 0: METADATA"),
        ("SECTION 1: FACTS", "SECTION 1: FACTS"),
        ("SECTION 2: ISSUES", "SECTION 2: ISSUES"),
        ("SECTION 3: RULES", "SECTION 3: RULES"),
        ("SECTION 4: APPLICATION", "SECTION 4: APPLICATION"),
        ("SECTION 5: CONCLUSION", "SECTION 5: CONCLUSION"),
        # Variations
        ("Section 0: Metadata", "Section 0: Metadata"),
        ("Section 1: Facts", "Section 1: Facts"),
        ("# SECTION 1: FACTS", "# SECTION 1: FACTS"),
        ("## SECTION 1: FACTS", "## SECTION 1: FACTS"),
    ]
    
    found_headers = []
    for pattern, label in section_patterns:
        if pattern in full_response:
            found_headers.append(label)
            print(f"✓ Found: {label}")
    
    if not found_headers:
        print("❌ No standard section headers found!")
        print("\nLooking for alternative patterns...")
        # Look for any section-like patterns
        import re
        section_like = re.findall(r'(?i)(?:section|part)\s*\d+[:\-]?\s*\w+', full_response[:1000])
        if section_like:
            print(f"Found section-like patterns: {section_like[:10]}")
    
    # Show extracted components preview
    print("\n" + "=" * 80)
    print("EXTRACTED COMPONENTS PREVIEW")
    print("=" * 80)
    
    for component in ['facts', 'issues', 'rules', 'application', 'conclusion']:
        content = result.get(component, '').strip()
        print(f"\n{component.upper()} ({len(content)} chars):")
        if content:
            print(content[:300])
            if len(content) > 300:
                print("... (truncated)")
        else:
            print("(EMPTY)")

