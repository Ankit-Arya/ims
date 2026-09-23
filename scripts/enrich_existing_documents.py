from __future__ import annotations

import argparse
from sqlalchemy import select

from ike.db.models import Chunk, Document
from ike.db.session import SessionLocal
from ike.services.metadata_enrichment import infer_operational_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich existing documents without OCR/chunk/embed work")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    updated = skipped = with_lines = with_rs = 0
    with SessionLocal() as db:
        docs = list(db.scalars(select(Document).order_by(Document.created_at)).all())
        if args.limit > 0:
            docs = docs[: args.limit]
        for document in docs:
            metadata = dict(document.extra_metadata or {})
            if metadata.get("operational_profile") and not args.force:
                skipped += 1
                continue
            samples = list(
                db.scalars(
                    select(Chunk.contextual_text)
                    .where(Chunk.document_id == document.id)
                    .order_by(Chunk.ordinal)
                    .limit(20)
                ).all()
            )
            profile = infer_operational_profile(
                document.title, document.original_filename, [x for x in samples if x]
            )
            with_lines += int(bool(profile.get("line_codes")))
            with_rs += int(bool(profile.get("rolling_stock")))
            metadata["operational_profile"] = profile
            if not args.dry_run:
                document.extra_metadata = metadata
            updated += 1
            if updated % 50 == 0:
                if not args.dry_run:
                    db.commit()
                print(f"processed={updated} skipped={skipped} with_lines={with_lines} with_rs={with_rs}", flush=True)
        if not args.dry_run:
            db.commit()
    print(f"complete processed={updated} skipped={skipped} with_lines={with_lines} with_rs={with_rs} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
