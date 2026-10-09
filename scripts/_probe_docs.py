from sqlalchemy import select
from ike.db.session import SessionLocal
from ike.db.models import Document, Chunk

with SessionLocal() as db:
    docs = db.scalars(select(Document).where(
        (Document.title.ilike("%MRGR%")) |
        (Document.original_filename.ilike("%MRGR%")) |
        (Document.title.ilike("%SOP-November%")) |
        (Document.original_filename.ilike("%SOP-November%")) |
        (Document.title.ilike("%A_DM%")) |
        (Document.original_filename.ilike("%A_DM%"))
    ).order_by(Document.title)).all()
    for d in docs:
        print("DOC", d.id, d.title, d.original_filename, d.ingestion_status, d.page_count)
        if "SOP-November" in (d.title or ""):
            hits=db.scalars(select(Chunk).where(
                Chunk.document_id==d.id,
                (Chunk.contextual_text.ilike("%VIKAS KUMAR%")) | (Chunk.text.ilike("%VIKAS KUMAR%"))
            ).order_by(Chunk.ordinal)).all()
            for c in hits[:5]:
                print(" SOP_HIT",c.page_from,c.ordinal,(c.contextual_text or c.text or "")[:800].replace("\n"," "))
        if "A_DM" in (d.title or ""):
            hits=db.scalars(select(Chunk).where(
                Chunk.document_id==d.id,
                ((Chunk.contextual_text.ilike("%SHAHDARA%")) | (Chunk.text.ilike("%SHAHDARA%"))) &
                ((Chunk.contextual_text.ilike("%8800793103%")) | (Chunk.text.ilike("%8800793103%")))
            ).order_by(Chunk.ordinal)).all()
            for c in hits[:5]:
                print(" ADM_HIT",c.page_from,c.ordinal,(c.contextual_text or c.text or "")[:1200].replace("\n"," "))
