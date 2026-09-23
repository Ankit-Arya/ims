# Security — 0.5.0 additions

Lazy visual endpoints reuse `document_access_clause(user)` before reading any PDF and accept only a short-lived signed token containing document/page/bbox metadata. The browser cannot submit a server filesystem path. Cached visual files remain behind authenticated routes; tokens expire. Final-answer/retrieval content is not cached across users in 0.5.0. Role-alias metadata caching is keyed by user ACL attributes, explicit document scope and accessible-corpus revision.

Nginx owns public 80/443. API, database, cache and inference diagnostics remain loopback/internal. TLS private keys are deployment secrets and are never included in the release ZIP.

---

## Historical / earlier-release notes


## Security boundary

IKE is an application-level internal knowledge platform. It enforces access through authenticated product routes and database predicates. It is not cryptographic isolation from operating-system/database administrators.

## Organisation document access

Organisation documents use `organization`, `department` and `restricted` ACL metadata. The document predicate is applied before retrieved text can become model evidence.

## Personal PDF isolation

Personal PDFs use a stricter ownership/share model:

```text
workspace_scope = personal
AND
(created_by = current_user OR explicit DocumentShare exists)
```

Admin status does not automatically satisfy that branch. This applies to normal:

- document metadata;
- source view/download;
- Q&A retrieval;
- selected-document retrieval;
- Deep Analysis source authorization.

Only the owner can change sharing or lifecycle-manage a personal PDF.

### Sharing directory

All authenticated users can query a minimal active-user directory containing user ID, username and department so they can select colleagues for sharing. If your organisation considers even that directory sensitive, restrict the endpoint by policy or integrate the corporate directory with appropriate visibility rules.

## Question/report privacy

- query history is filtered by `query_logs.user_id == current_user.id`;
- feedback can only target the current user's query;
- Deep Analysis jobs are filtered by `report_jobs.created_by == current_user.id`;
- earlier Q&A is never inserted into later prompt context.

## Source files

View/download endpoints first apply document authorization and only then return the server-known storage path. Clients cannot provide arbitrary filesystem paths.

Delete paths are derived from server-side document IDs and recorded storage metadata.

## Ingestion

Only PDF uploads are accepted in this release. File-size and PDF validation are enforced before indexing.

Docling model artifacts are prefetched into persistent storage so user ingestion does not normally require arbitrary internet access.

## Prompt injection

Retrieved document text is untrusted evidence. IKE does not grant retrieved text authority to call arbitrary external tools or mutate systems. Prompt injection remains an unsolved model-security problem, so sensitive deployments should additionally:

- classify source documents;
- scan/monitor suspicious instruction-like content;
- maintain deterministic tool permissions;
- avoid giving normal Q&A write access to enterprise systems;
- perform red-team testing against the actual corpus.

## Browser security

The built-in UI uses an HttpOnly `SameSite=Lax` session cookie. For exposure beyond a tightly controlled internal origin, add your organisation's standard:

- TLS termination;
- explicit CSRF protection;
- Content Security Policy;
- security headers;
- centralized identity/SSO if required;
- rate limiting and audit retention.

Theme preference is stored in browser localStorage and contains no document/question data.

## Administrator expectations

Product UI/API privacy does not prevent an infrastructure administrator with direct database/filesystem access from inspecting records. If the requirement becomes “even platform administrators cannot read a user's private PDFs,” a different encrypted tenant-key architecture is required.

## Security acceptance tests

Before broad rollout, test at least:

1. User A personal PDF is inaccessible to User B by list, UUID metadata, view, download, Q&A selection and Deep Analysis selection.
2. Sharing with User B grants those product capabilities.
3. Removing User B from shares revokes access immediately for new requests.
4. Unshared admin cannot access a personal PDF through normal product endpoints.
5. Department/restricted organisation ACLs still work.
6. User A question/report history is absent for User B.
7. Malformed/oversized uploads are rejected.
8. Direct object-reference guessing fails closed.
9. Deleted PDFs no longer participate in retrieval.
10. Dependency/container/SAST/secret scans are part of deployment CI.