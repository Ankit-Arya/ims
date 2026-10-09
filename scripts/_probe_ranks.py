from sqlalchemy import select
from ike.db.session import SessionLocal
from ike.db.models import User
from ike.agent.tools import CorpusTools

tests = [
    ("MD", "current Managing Director of Delhi Metro Rail Corporation (DMRC)", ["DMRC", "Managing Director"], "SOP-November-2025.pdf"),
    ("SHAHDARA", "Shahdara Metro Station contact number telephone station control room", ["Shahdara", "contact number", "station"], "A_DM_Rev.01_.pdf"),
    ("WELCOME", "Welcome metro station mobile number", ["Welcome", "mobile number", "station"], "A_DM_Rev.01_.pdf"),
    ("CIVIL", "contact number of Civil Line metro station", ["Civil Line", "contact number", "station"], "A_DM_Rev.01_.pdf"),
]
with SessionLocal() as db:
    user=db.scalar(select(User).where(User.is_active.is_(True)).order_by(User.created_at).limit(1))
    tools=CorpusTools(db,user,request_id="rank-probe")
    for label,q,terms,target in tests:
        print("\n###",label)
        r=tools.search(q,mode="hybrid",exact_terms=terms,document_ids=[],top_k=24)
        print("META",r.metadata)
        found=[]
        for i,item in enumerate(r.items,1):
            if item.get("document_title")==target:
                found.append((i,item.get("page_from"),(item.get("snippet") or "")[:500].replace("\n"," ")))
        print("TARGET",found)
        print("TOP",[(i,x.get("document_title"),x.get("page_from")) for i,x in enumerate(r.items[:12],1)])
