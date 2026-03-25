import asyncio

import pytest

from memory_module.storage.local_storage import LocalStorage
from tests.conftest import make_upload_file


def test_local_storage_save_read_delete(tmp_path):
    storage = LocalStorage(str(tmp_path))
    upload = make_upload_file("note.txt", b"hello", "text/plain")

    saved_path = asyncio.run(storage.save(upload, "nested/note.txt"))
    assert saved_path.endswith("nested/note.txt")

    content = asyncio.run(storage.read("nested/note.txt"))
    assert content == b"hello"

    asyncio.run(storage.delete("nested/note.txt"))
    with pytest.raises(FileNotFoundError):
        asyncio.run(storage.read("nested/note.txt"))


def test_local_storage_rejects_path_traversal(tmp_path):
    storage = LocalStorage(str(tmp_path))
    upload = make_upload_file("note.txt", b"hello", "text/plain")

    with pytest.raises(ValueError, match="outside the configured base path"):
        asyncio.run(storage.save(upload, "../escape.txt"))


def test_local_storage_resets_stream_after_save(tmp_path):
    storage = LocalStorage(str(tmp_path))
    upload = make_upload_file("note.txt", b"hello", "text/plain")

    asyncio.run(storage.save(upload, "note.txt"))

    assert upload.file.tell() == 0
    assert upload.file.read() == b"hello"


def test_local_storage_missing_delete_raises(tmp_path):
    storage = LocalStorage(str(tmp_path))

    with pytest.raises(FileNotFoundError):
        asyncio.run(storage.delete("missing.txt"))

