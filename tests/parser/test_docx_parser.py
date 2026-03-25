from memory_module.parser.docx_parser import DocxParser
from tests.conftest import build_docx_bytes, make_upload_file


def test_accepts_valid_docx(upload_docx):
    parser = DocxParser()

    assert parser.accepts(upload_docx) is True
    assert parser.last_error is None


def test_accepts_invalid_docx_sets_last_error():
    parser = DocxParser()
    upload = make_upload_file(
        "bad.docx",
        b"broken",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    assert parser.accepts(upload) is False
    assert "corrupt" in parser.last_error


def test_convert_returns_text_and_file_metadata(upload_docx):
    parser = DocxParser()

    result = parser.convert(upload_docx)

    assert "First paragraph" in result.text
    assert "Second paragraph" in result.text
    assert result.file_metadata.document_title == "sample"
    assert result.file_metadata.document_id.startswith("doc_")


def test_convert_document_id_is_deterministic_for_same_file(docx_bytes):
    parser = DocxParser()
    first = make_upload_file(
        "same.docx",
        docx_bytes,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    second = make_upload_file(
        "same.docx",
        docx_bytes,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    assert parser.convert(first).file_metadata.document_id == parser.convert(second).file_metadata.document_id


def test_convert_document_id_changes_for_different_content():
    parser = DocxParser()
    first = make_upload_file(
        "first.docx",
        build_docx_bytes(["one"]),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    second = make_upload_file(
        "second.docx",
        build_docx_bytes(["two"]),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    assert parser.convert(first).file_metadata.document_id != parser.convert(second).file_metadata.document_id

