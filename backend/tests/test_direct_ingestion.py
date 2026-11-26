from build_case_chunks import _is_header_block
from embed_knowledge import embed_chunks
from legal_text import pdf_ligature_map


def test_boundary_filter_retains_substantive_text():
    assert not _is_header_block("Certificate must accompany the record.", 6, 15, 800)
    assert _is_header_block("Page 1 of 2", 786, 798, 800)
    assert not _is_header_block("22", 120, 150, 800)


def test_glyph_repairs_are_specific_to_document_encoding():
    assert "certi\ue003cate".translate(pdf_ligature_map("certi\ue003cate")) == "certificate"
    assert "a\ue003davit".translate(pdf_ligature_map("a\ue003davit")) == "affidavit"
    assert pdf_ligature_map("certi\ue003cate and a\ue003davit") == {}


def test_embedding_updates_changed_text_once():
    class Collection:
        rows = {}
        name = "cases"
        def get(self, include):
            return {"ids": list(self.rows), "metadatas": list(self.rows.values())}
        def upsert(self, ids, metadatas, **kwargs):
            self.rows.update(zip(ids, metadatas))
    class Model:
        def encode(self, texts, **kwargs):
            class Vector(list):
                def tolist(self):
                    return list(self)
            return Vector([[1]] * len(texts))
    col = Collection()
    model = Model()
    chunks = [{"chunk_id": "case_0", "text": "Prior ruling."}]
    assert embed_chunks(chunks, col, model, "cases", verbose=False) == 1
    assert embed_chunks(chunks, col, model, "cases", verbose=False) == 0
    chunks[0]["text"] = "Revised ruling."
    assert embed_chunks(chunks, col, model, "cases", verbose=False) == 1
    assert len(col.rows) == 1
