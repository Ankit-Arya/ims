from sqlalchemy import select
from ike.db.session import SessionLocal
from ike.db.models import User
from ike.agent.tools import CorpusTools

with SessionLocal() as db:
    user=db.scalar(select(User).where(User.is_active.is_(True)).order_by(User.created_at).limit(1))
    tools=CorpusTools(db,user,request_id="civil-probe")
    for term in ["Civil Lines", "Mobile No", "Civil Lines Metro Station"]:
        print("\nTERM",term)
        rows=tools.search_engine.table_exact(term,user,None,120)
        for i,c in enumerate(rows,1):
            if c.document_title=="A_DM_Rev.01_.pdf":
                print("A_DM",i,c.page_from,c.ordinal,(c.contextual_text or c.text or "")[:900].replace("\n"," "))
    rows=tools.search_engine.table_relaxed_lexical("Civil Lines Metro Station contact number mobile number",user,None,120)
    print("\nRELAXED A_DM")
    for i,c in enumerate(rows,1):
        if c.document_title=="A_DM_Rev.01_.pdf":
            print(i,c.page_from,c.ordinal,(c.contextual_text or c.text or "")[:900].replace("\n"," "))
