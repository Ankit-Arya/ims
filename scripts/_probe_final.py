from sqlalchemy import select
from ike.db.session import SessionLocal
from ike.db.models import User
from ike.agent.tools import CorpusTools

cases = [
    ("MRGR_SOURCE", "Metro Railways General Rules 2020 MRGR book Rule 31", [], "02. MRGR 2020(1).pdf", "source"),
    ("MRGR_RULE31", "what is mentioned in Rule number 31 of MRGR 2020 book", ["MRGR 2020", "Rule 31", "Examination of trains"], "02. MRGR 2020(1).pdf", "search"),
    ("SHAHDARA", "Shahdara Metro Station contact number mobile number", ["Shahdara", "Mobile No", "Mobile Number", "contact number"], "A_DM_Rev.01_.pdf", "search"),
    ("WELCOME", "Welcome Metro Station contact number mobile number", ["Welcome", "Mobile No", "Mobile Number", "contact number"], "A_DM_Rev.01_.pdf", "search"),
    ("CIVIL", "Civil Lines Metro Station contact number mobile number", ["Civil Lines", "Mobile No", "Mobile Number", "contact number"], "A_DM_Rev.01_.pdf", "search"),
    ("ALL_STATIONS", "station directory contact telephone mobile numbers station name", ["Station Name", "Mobile No", "Mobile Number", "Contact Number"], "A_DM_Rev.01_.pdf", "enumerate"),
]
with SessionLocal() as db:
    user=db.scalar(select(User).where(User.is_active.is_(True)).order_by(User.created_at).limit(1))
    tools=CorpusTools(db,user,request_id="final-probe")
    for label,q,terms,target,kind in cases:
        print("\n###",label)
        if kind=="source":
            r=tools.search_documents(q)
            rows=r.items
            for i,x in enumerate(rows,1):
                if x.get("title")==target:
                    print("TARGET_RANK",i,x.get("score"),x.get("metadata_hints"),x.get("matched_sections"))
            print("TOP5",[(i,x.get("title"),x.get("score")) for i,x in enumerate(rows[:5],1)])
            continue
        if kind=="enumerate":
            r=tools.enumerate(q,mode="hybrid",exact_terms=terms,document_ids=[],top_k=48)
        else:
            r=tools.search(q,mode="hybrid",exact_terms=terms,document_ids=[],top_k=14)
        hits=[]
        for i,x in enumerate(r.items,1):
            if x.get("document_title")==target:
                hits.append((i,x.get("page_from"),x.get("direct_score"),(x.get("snippet") or "")[:550].replace("\n"," ")))
        print("TARGET",hits[:8])
        print("LANES",r.metadata.get("lane_status"))
        print("TOP8",[(i,x.get("document_title"),x.get("page_from"),x.get("direct_score")) for i,x in enumerate(r.items[:8],1)])
