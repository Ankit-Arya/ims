# User Guide — 0.5.0

## Ask naturally

Users do not need to know exact manual wording. Ask the information need in ordinary language. Auto preserves literal terms, uses corpus-supported terminology/aliases and chooses a bounded or broader retrieval path. Answers lead with the direct result and, when evidence supports it, proactively add useful applicability, prerequisites, conditions, limits, exceptions, alternative scenarios and immediate consequences. IMS does not fill missing documentary facts from general knowledge.

## Workspace

Desktop uses a left control/navigation rail and a wide answer workspace. Mode, Sources and PDF selection are on the left. The question composer stays at the bottom. User activity and recent questions open in a drawer so technical answers, tables and figures keep the width of the page. Knowledge and My PDFs retain searchable scrollable PDF lists.

When retrieved text references a useful figure/diagram/table, IMS can lazily render the authorised source page/crop below the answer. Click the visual to open it larger.

---

## Historical / earlier-release notes


## Workspace layout

On desktop IMS uses three working areas:

- **Left rail** — Ask IMS, My Questions, Knowledge, My PDFs and (for administrators) Users.
- **Centre** — the current working page.
- **Right rail** — signed-in user details, question/PDF activity counts and recent questions.

On narrower screens the right rail is removed and navigation compresses responsively.

## Ask IMS

The Ask page is designed like a modern AI workspace while preserving IMS's document-grounded behavior.

1. Type the question in the composer at the bottom of the page.
2. Select **Auto**, **Direct**, **Research** or **Deep Analysis** inside the composer.
3. Select the source scope: **All available**, **Knowledge**, **My PDFs** or **Selected PDFs**.
4. Use the **PDFs** control when you want a specific subset. The picker itself has PDF search.
5. Press **Enter** to submit. Use **Shift+Enter** for a new line.

While the request runs, completed workflow stages remain visible and the conversation area follows the active stage. When the final answer is returned, IMS brings the start of the answer into view.

Questions submitted in the same browser session remain visible in the Ask transcript for convenience. They are still **independent requests**: prior questions and answers are not added to later retrieval or LLM prompts.

### Answer modes

| Mode | Use when | Typical example |
| --- | --- | --- |
| **Auto** | You do not want to choose. IMS routes focused questions efficiently and broad questions to Research. | `What is BIC?` |
| **Direct** | A focused factual answer is needed quickly. | `What is the maximum speed in RM mode?` |
| **Research** | Breadth/completeness across evidence matters. | `Find all conditions and exceptions for door isolation.` |
| **Deep Analysis** | You want a long, structured review across selected PDFs. | `Compare these three manuals and identify differences, conflicts and operational risks.` |

Deep Analysis is asynchronous and can take materially longer than ordinary Q&A because it analyses document sections rather than only top-ranked passages.

## My Questions

Contains your Q&A and Deep Analysis history. Use the search field to filter prior work.

This is history, **not conversational memory**.

## Knowledge

The organisation document library available to your account.

The document panel now remains bounded to the page. Use:

- **Search Knowledge PDFs** to match title, filename and indexed document metadata shown by the application;
- **status filter** for Ready, Indexing or Failed;
- the internal scroll area to move through large libraries without stretching the entire page.

Permitted users can view/download source PDFs. Admins and authorised analyst uploaders retain retry, reprocess and delete controls.

## My PDFs

Every authenticated user has a personal PDF library. The page has the same search, status filtering and bounded scrolling behavior as Knowledge.

A personal PDF is private by default. If you own it, **Manage access** lets you share it with selected active users.

A shared user may view/download the PDF and include it in Q&A, Research and Deep Analysis. A shared user cannot change sharing or delete/reprocess the owner's PDF.

Administrators do not automatically inherit access to another user's personal PDFs through normal product routes.

## Activity rail

The right-side activity rail shows:

- total Q&A questions asked by the signed-in user;
- PDFs currently available to that user;
- ready PDFs;
- PDFs currently queued/processing;
- the five most recent Q&A questions.

The counts are user-scoped and respect normal PDF access controls.

## Theme

Use the moon/sun control in the top bar to switch between light and dark themes. The choice is stored in that browser.

## User roles

### user

- ask/search accessible Knowledge and My PDFs;
- upload/manage own personal PDFs;
- share own personal PDFs;
- view/download accessible PDFs;
- run Deep Analysis;
- view only own question/analysis history.

### analyst

Everything above, plus upload organisation Knowledge documents and manage organisation documents they uploaded.

### admin

Organisation corpus administration and account management. Personal-document product access still requires ownership or explicit sharing.