from __future__ import annotations

import gzip
import io
import tarfile
from pathlib import Path

import pytest

from product_search.release_build import normalize_sdist


def _write_sdist(path: Path, *, mtime: int, link: bool = False) -> None:
    with tarfile.open(path, "w:gz") as archive:
        root = tarfile.TarInfo("pkg-1.0")
        root.type = tarfile.DIRTYPE
        root.mtime = mtime
        archive.addfile(root)
        if link:
            member = tarfile.TarInfo("pkg-1.0/link")
            member.type = tarfile.SYMTYPE
            member.linkname = "/tmp/outside"
            member.mtime = mtime
            archive.addfile(member)
        else:
            data = b"hello\n"
            member = tarfile.TarInfo("pkg-1.0/file.txt")
            member.size = len(data)
            member.mtime = mtime
            member.mode = 0o600
            archive.addfile(member, io.BytesIO(data))


def test_normalized_sdist_is_byte_reproducible(tmp_path: Path):
    first_raw = tmp_path / "first.tar.gz"
    second_raw = tmp_path / "second.tar.gz"
    _write_sdist(first_raw, mtime=1000)
    _write_sdist(second_raw, mtime=2000)
    first = tmp_path / "first-normalized.tar.gz"
    second = tmp_path / "second-normalized.tar.gz"
    normalize_sdist(first_raw, first, source_date_epoch=1_577_836_800)
    normalize_sdist(second_raw, second, source_date_epoch=1_577_836_800)
    assert first.read_bytes() == second.read_bytes()
    assert int.from_bytes(first.read_bytes()[4:8], "little") == 1_577_836_800
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(first.read_bytes())), mode="r:") as archive:
        assert {int(member.mtime) for member in archive.getmembers()} == {1_577_836_800}


def test_normalized_sdist_rejects_links(tmp_path: Path):
    raw = tmp_path / "links.tar.gz"
    _write_sdist(raw, mtime=1000, link=True)
    with pytest.raises(RuntimeError, match="must not contain links"):
        normalize_sdist(raw, tmp_path / "normalized.tar.gz", source_date_epoch=1_577_836_800)
