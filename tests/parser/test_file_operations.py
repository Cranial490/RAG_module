from io import BytesIO

import pytest

from memory_module.parser.file_operations import FileOps
from tests.conftest import build_docx_bytes, make_upload_file


def test_validate_accepts_valid_docx(docx_bytes):
    file_ops = FileOps()
    upload = make_upload_file(
        "ok.docx",
        docx_bytes,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    assert file_ops.validate(upload, allowed_extensions=[".docx"]) is True


def test_validate_rejects_invalid_extension(docx_bytes):
    file_ops = FileOps()
    upload = make_upload_file("bad.txt", docx_bytes, "text/plain")

    with pytest.raises(ValueError, match="not allowed"):
        file_ops.validate(upload, allowed_extensions=[".docx"])


def test_validate_rejects_mime_mismatch(docx_bytes):
    file_ops = FileOps()
    upload = make_upload_file("bad.docx", docx_bytes, "application/pdf")

    with pytest.raises(ValueError, match="does not match expected type"):
        file_ops.validate(upload, allowed_extensions=[".docx"])


def test_validate_rejects_empty_file():
    file_ops = FileOps()
    upload = make_upload_file(
        "empty.docx",
        b"",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    with pytest.raises(ValueError, match="empty"):
        file_ops.validate(upload, allowed_extensions=[".docx"])


def test_validate_rejects_file_over_size_limit(docx_bytes):
    file_ops = FileOps()
    upload = make_upload_file(
        "large.docx",
        docx_bytes,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    with pytest.raises(ValueError, match="maximum allowed size"):
        file_ops.validate(upload, allowed_extensions=[".docx"], max_file_size_bytes=10)


def test_validate_rejects_corrupt_docx():
    file_ops = FileOps()
    upload = make_upload_file(
        "corrupt.docx",
        b"not-a-real-docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    with pytest.raises(ValueError, match="corrupt"):
        file_ops.validate(upload, allowed_extensions=[".docx"])


def test_validate_text_requires_utf8():
    file_ops = FileOps()
    upload = make_upload_file("bad.txt", b"\xff\xfe", "text/plain")

    with pytest.raises(ValueError, match="UTF-8"):
        file_ops.validate(upload, allowed_extensions=[".txt"])


def test_validate_pdf_checks_signature_and_eof():
    file_ops = FileOps()
    bad_pdf = make_upload_file("bad.pdf", b"not-pdf", "application/pdf")

    with pytest.raises(ValueError, match="corrupt"):
        file_ops.validate(bad_pdf, allowed_extensions=[".pdf"])

    ok_pdf = make_upload_file("ok.pdf", b"%PDF-1.7\nbody\n%%EOF", "application/pdf")
    assert file_ops.validate(ok_pdf, allowed_extensions=[".pdf"]) is True


def test_compute_file_hash_is_deterministic(docx_bytes):
    file_ops = FileOps()
    first = make_upload_file(
        "sample.docx",
        docx_bytes,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    second = make_upload_file(
        "sample.docx",
        docx_bytes,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    assert file_ops.compute_file_hash(first) == file_ops.compute_file_hash(second)

