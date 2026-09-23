import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from fastapi import UploadFile

from ike.core.config import get_settings

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class UploadValidationError(ValueError):
    """Base class for local upload validation failures."""


class UploadTooLargeError(UploadValidationError):
    """Raised when an upload exceeds the configured byte limit."""


class InvalidPdfError(UploadValidationError):
    """Raised when a file does not contain a PDF header near the start."""


@dataclass(slots=True)
class StoredFile:
    path: Path
    sha256: str
    size_bytes: int
    original_filename: str


class LocalStorage:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.settings.originals_dir.mkdir(parents=True, exist_ok=True)
        self.settings.parsed_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def safe_name(name: str) -> str:
        cleaned = _SAFE.sub("_", Path(name).name).strip("._")
        return cleaned[:180] or "document.pdf"

    def save_upload(self, upload: UploadFile, document_id: UUID) -> StoredFile:
        filename = self.safe_name(upload.filename or "document.pdf")
        target_dir = self.settings.originals_dir / str(document_id)
        target_dir.mkdir(parents=True, exist_ok=False)
        target = target_dir / filename
        digest = hashlib.sha256()
        size = 0
        limit = self.settings.max_upload_mb * 1024 * 1024
        with target.open("wb") as handle:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    handle.close()
                    target.unlink(missing_ok=True)
                    target_dir.rmdir()
                    raise UploadTooLargeError(f"File exceeds {self.settings.max_upload_mb} MB upload limit")
                digest.update(chunk)
                handle.write(chunk)
        with target.open("rb") as handle:
            header = handle.read(1024)
        if b"%PDF-" not in header:
            target.unlink(missing_ok=True)
            target_dir.rmdir()
            raise InvalidPdfError("Uploaded file does not contain a valid PDF header")
        return StoredFile(target, digest.hexdigest(), size, upload.filename or filename)

    def parsed_json_path(self, document_id: UUID) -> Path:
        directory = self.settings.parsed_dir / str(document_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / "canonical_document.json"

    def purge_document_files(self, document_id: UUID) -> None:
        """Delete source and derived files using only server-derived directories.

        Never accepts a caller-provided filesystem path. This keeps deletion bounded to
        the configured originals/parsed roots even if a database path were malformed.
        """

        for root in (self.settings.originals_dir, self.settings.parsed_dir):
            target = root / str(document_id)
            if target.exists():
                shutil.rmtree(target)
