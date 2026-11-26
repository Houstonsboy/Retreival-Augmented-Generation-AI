"""Conservative repairs for PDF glyphs whose Unicode mapping was lost."""
import re

_LIGATURE_ANCHORS = (
    (r"\bcerti([\ue000-\uf8ff])cate", "fi"),
    (r"\ba([\ue000-\uf8ff])davit", "ffi"),
    (r"\bo([\ue000-\uf8ff])ce\b", "ffi"),
    (r"\bdi([\ue000-\uf8ff])cult\b", "ffi"),
    (r"\bplainti([\ue000-\uf8ff])", "ff"),
    (r"\bin([\ue000-\uf8ff])uence", "fl"),
)

def pdf_ligature_map(text):
    candidates = {}
    for pattern, replacement in _LIGATURE_ANCHORS:
        for glyph in re.findall(pattern, text, re.IGNORECASE):
            candidates.setdefault(glyph, set()).add(replacement)
    return {ord(glyph): next(iter(values)) for glyph, values in candidates.items() if len(values) == 1}
