# Compositional Q&A capability model — IMS 0.7.0

IMS is an evidence-bound internal-document system, not a general world-knowledge chatbot. “Can handle” below means the workflow has an explicit mechanism to retrieve and verify the information need from accessible indexed documents; it does not mean every corpus necessarily contains the answer.

| Question class | 0.7.0 behavior |
|---|---|
| Single acronym/term | Existing lookup path retained; distinct supported meanings can survive. |
| Several acronyms/identifiers | Independent definition goals and lookup/search branches. |
| Several ordinary-language terms | Fast semantic planner can decide whether they are separate concepts or one compound request. |
| Multiple requested attributes | One requirement per entity/facet; goal-balanced evidence. |
| Lists/enumerations | Existing entity coverage retained; generic document-diverse coverage is available for enumeration/overview goals. |
| Compare A/B/C | Independent entity evidence plus a mandatory comparison goal. |
| Relationship / “does A affect B?” | Relationship itself must be supported; definitions/co-occurrence are insufficient. |
| Yes/no / “right?” / possibly wrong premise | Claim-check planning and verification; contradiction can be a valid resolved result. |
| Conditional “if/when/unless” | Semantic planner decomposes conditions/outcomes; qualifiers remain visible to the audit and answer. |
| Multi-event / multi-hop scenario | Atomic goals, dependencies, independent retrieval and targeted recovery. |
| Exceptions / negation | Planner preserves negation/exception wording; verifier checks lost conditions. |
| Cross-document synthesis | Allowed when evidence supplies compatible premises/links; unsupported joins must be stated as unestablished. |
| Conflicting sources/revisions | Existing revision/authority metadata retained; synthesis/verification must surface conflict rather than silently blend it. |
| Current/latest/revision question | Temporal goal can be planned; precedence may be asserted only from effective/revision/authority evidence. |
| Calculations based on policy values | Calculation goal is planned; answer must identify evidenced inputs/formula/assumptions and cannot invent missing inputs. |
| Broad “everything about …” | Bounded overview strategy and document-diverse coverage; no unbounded corpus dump. |
| Typos / indirect wording | Semantic planner can create retrieval hypotheses while sanitized seed planning preserves user identifiers/numbers. |
| Explicit document scope | Existing ACL/document scope applies to every branch. |
| No-answer question | Missing goal remains visible; targeted recovery runs once, then answer scopes the unresolved part. |
| Confirmation/leading premise (“…, right/correct?”) | Treated as a claim to verify; the user's wording is not promoted into evidence. |
| Conflicting manuals / “which applies?” | Conflict/precedence is a required relationship goal; resolution needs revision/authority/effective evidence rather than score alone. |
| Mixed multi-intent request | Semantic planner can create separate definition, attribute, relation, condition and outcome goals; required user clauses are re-added if the planner drops them. |
| Unicode / non-English wording | Planner tokenization preserves Unicode words and routes multiword indirect wording to semantic planning rather than assuming English-only tokens. |
| Planner/audit failure | Semantic-planner failure forces a safer Research/verification fallback; an audit cannot mark a goal supported with evidence attributed only to another goal. |
| Overview-only evidence | Overview sources are searched once per atomic goal in compositional retrieval so later goals cannot disappear merely because overview documents are isolated from the operational lane. |
| Recovery when initial context is full | Recovered evidence is admitted before the existing bounded evidence context, preventing a full initial context from discarding the very evidence recovery found. |
| Corpus-wide negative complement (“which documents do not mention X?”) | **Conservative boundary:** top-k absence is never treated as proof. The request is explicitly flagged as requiring exhaustive inventory; until such an operation exists, IMS must not fabricate a negative document list. |
| External/live facts not in indexed corpus | Out of scope for document-grounded Q&A unless a separate approved connector is added. |
| Purely subjective/advisory question with no governing document | IMS may report lack of documentary support; it must not manufacture organizational policy. |
| User asks to ignore the documents / answer from memory | System remains evidence-bound; user wording cannot relax source-grounding rules. |
| Pronoun-only follow-up (“where is it?”, “what about that?”) | **Conservative boundary:** current persistent context safely carries explicit line/rolling-stock scope, not arbitrary prior answer facts. If the referent is not explicit/resolvable, IMS should not guess it. |

## Why there is still a boundary

No finite RAG system can guarantee a correct answer to every arbitrary natural-language question. The engineering target is instead:

1. represent the user's information need without silently dropping parts;
2. retrieve each required part independently;
3. distinguish relevance from proof;
4. detect missing/contradicted requirements before drafting;
5. prevent evidence for one goal from being borrowed as proof for another;
6. recover narrowly rather than globally increasing top-k;
7. abstain or scope uncertainty when the corpus cannot establish the required relation/fact.

This failure-safe behavior is preferable to a fluent answer that silently answers only the easiest part of a complex request.
