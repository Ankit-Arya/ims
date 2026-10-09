from sqlalchemy import select
from ike.db.session import SessionLocal
from ike.db.models import User
from ike.agent.tools import CorpusTools

terms=["Civil Lines","Mobile No","Mobile Number","contact number"]
with SessionLocal() as db:
    user=db.scalar(select(User).where(User.is_active.is_(True)).order_by(User.created_at).limit(1))
    tools=CorpusTools(db,user,request_id="section-prior-probe")
    lanes=[]
    byid={}
    for idx,term in enumerate(terms,1):
        vals=tools.search_engine.table_exact(term,user,None,120)
        lanes.append((f"table_exact_term_{idx}",vals,1.0))
        for c in vals: byid[c.chunk_id]=c
    prior=tools._table_section_prior_ids(lanes)
    for i,cid in enumerate(prior[:80],1):
        c=byid[cid]
        if c.document_title=="A_DM_Rev.01_.pdf" or i<=25:
            print(i,c.document_title,c.page_from,c.ordinal,c.section_path,(c.contextual_text or c.text or "")[:240].replace("\n"," "))
