"""Archive I/O for the data-acquisition notebook; no model code."""

from pathlib import Path
import hashlib
import json
import zipfile


def file_sha(path):
    """Hash the entire file without reading it into RAM."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def process_archive(item, P, SOURCE, DOWNLOAD, completed, completed_path):
    """Reuse a verified extraction or verify and extract one pinned archive.

    The manifest is updated only after successful extraction. The compressed
    archive is then removed; raw feature JSONL files and checksums remain.
    """
    from huggingface_hub import hf_hub_download

    previous = completed["archives"].get(item["filename"])
    if previous and previous["sha256"] == item["sha256"]:
        valid = True
        for entry in previous["extracted"]:
            dest = P.data / entry["file"]
            if not dest.exists() or dest.stat().st_size != entry["bytes"]:
                valid = False
                break
            if (
                dest.stat().st_mtime_ns != entry["mtime_ns"]
                and file_sha(dest) != entry["sha256"]
            ):
                valid = False
                break
        if valid:
            print("Reusing verified extracted archive:", item["filename"], flush=True)
            return
    print(
        "Downloading",
        item["filename"],
        f"({item['bytes'] / 1000000000.0:.2f} GB)",
        flush=True,
    )
    archive = Path(
        hf_hub_download(
            repo_id=SOURCE["repo_id"],
            repo_type="dataset",
            revision=SOURCE["revision"],
            filename=item["filename"],
            local_dir=str(DOWNLOAD),
        )
    )
    assert archive.stat().st_size == item["bytes"]
    assert file_sha(archive) == item["sha256"], "Archive SHA256 mismatch"
    print("Archive checksum passed; extracting", item["filename"], flush=True)
    # Extract data-only members using their base names, never archive paths.
    extracted = []
    with zipfile.ZipFile(archive) as z:
        members = [
            info
            for info in z.infolist()
            if not info.is_dir() and info.filename.endswith(".jsonl")
        ]
        assert members, "No feature JSONL files in archive"
        assert len({Path(info.filename).name for info in members}) == len(members)
        for info in members:
            dest = P.data / Path(info.filename).name
            assert "Win64" in dest.name
            partial = dest.with_suffix(".jsonl.partial")
            h = hashlib.sha256()
            lines = 0
            last = b""
            with z.open(info) as source, partial.open("wb") as target:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    target.write(chunk)
                    h.update(chunk)
                    lines += chunk.count(b"\n")
                    last = chunk[-1:]
            assert partial.stat().st_size == info.file_size
            if last and last != b"\n":
                lines += 1
            partial.replace(dest)
            extracted.append(
                {
                    "file": dest.name,
                    "bytes": info.file_size,
                    "sha256": h.hexdigest(),
                    "raw_rows": lines,
                    "mtime_ns": dest.stat().st_mtime_ns,
                }
            )
    completed["archives"][item["filename"]] = {
        "sha256": item["sha256"],
        "bytes": item["bytes"],
        "extracted": extracted,
    }
    completed_path.write_text(json.dumps(completed, indent=2))
    archive.unlink()
    print(
        "Completed",
        item["filename"],
        "files:",
        len(extracted),
        "raw rows:",
        sum((f["raw_rows"] for f in extracted)),
        flush=True,
    )
