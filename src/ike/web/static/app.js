let me = null;
let orgDocs = [];
let personalDocs = [];
let directoryUsers = [];
let selectedMode = 'research';
let historyItems = [];
let documentPollTimer = null;
let activeShareDocumentId = null;
let dashboardSummary = null;
let selectedQueryDocumentIds = new Set();
let activeQaRequest = null;
let activeRealtimeRequest = null;
let realtimeContext = null;
const answerTextByCard = new WeakMap();
let sourcePreviewTimer = null;

const $ = id => document.getElementById(id);
const allDocs = () => [...orgDocs, ...personalDocs];
const readyDocs = () => allDocs().filter(doc => doc.ingestion_status === 'ready');
const reportMaxDocuments = Number(document.body.dataset.reportMaxDocuments || 20);

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}

function validationMessage(detail) {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map(item => {
    const field = Array.isArray(item.loc) ? item.loc.filter(x => x !== 'body').join('.') : '';
    return `${field ? field + ': ' : ''}${item.msg || 'Invalid value'}`;
  }).join(' · ');
  return JSON.stringify(detail || 'Request failed');
}

async function api(path, opts = {}) {
  const response = await fetch(path, {credentials:'same-origin', ...opts});
  if (response.status === 204) return null;
  const raw = await response.text();
  let data = null;
  if (raw) { try { data = JSON.parse(raw); } catch { data = {detail: raw}; } }
  if (!response.ok) throw new Error(validationMessage(data?.detail || data || `Request failed (${response.status})`));
  return data;
}

function inlineMarkdown(text) {
  let value = escapeHtml(text);
  value = value.replace(/`([^`]+)`/g, '<code>$1</code>');
  value = value.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  value = value.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  value = value.replace(/\[(E\d+|R\d{5})\]/g, '<span class="inline-citation" data-evidence-id="$1" tabindex="0" aria-label="Preview source $1">[$1]</span>');
  return value;
}

function splitTableRow(line) {
  return line.trim().replace(/^\|/,'').replace(/\|$/,'').split('|').map(value => value.trim());
}

function renderMarkdown(markdown) {
  const lines = String(markdown || '').replace(/\r/g,'').split('\n');
  const out = [];
  let listType = null;
  let inCode = false;
  let codeLines = [];
  const closeList = () => { if (listType) out.push(`</${listType}>`); listType = null; };
  for (let index = 0; index < lines.length; index++) {
    const raw = lines[index];
    const line = raw.trimEnd();
    if (line.trim().startsWith('```')) {
      closeList();
      if (!inCode) { inCode = true; codeLines = []; }
      else { out.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`); inCode = false; }
      continue;
    }
    if (inCode) { codeLines.push(raw); continue; }
    if (line.includes('|') && index + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[index + 1])) {
      closeList();
      const headers = splitTableRow(line);
      index += 2;
      const rows = [];
      while (index < lines.length && lines[index].trim() && lines[index].includes('|')) {
        rows.push(splitTableRow(lines[index]));
        index++;
      }
      index--;
      out.push('<div class="table-wrap"><table><thead><tr>' + headers.map(header => `<th>${inlineMarkdown(header)}</th>`).join('') + '</tr></thead><tbody>');
      rows.forEach(row => out.push('<tr>' + row.map(cell => `<td>${inlineMarkdown(cell)}</td>`).join('') + '</tr>'));
      out.push('</tbody></table></div>');
      continue;
    }
    if (!line.trim()) { closeList(); continue; }
    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) { closeList(); const level = Math.min(4, heading[1].length + 1); out.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`); continue; }
    const bullet = line.match(/^\s*[-*]\s+(.+)$/);
    if (bullet) { if (listType !== 'ul') { closeList(); listType = 'ul'; out.push('<ul>'); } out.push(`<li>${inlineMarkdown(bullet[1])}</li>`); continue; }
    const numbered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (numbered) { if (listType !== 'ol') { closeList(); listType = 'ol'; out.push('<ol>'); } out.push(`<li>${inlineMarkdown(numbered[1])}</li>`); continue; }
    closeList();
    out.push(`<p>${inlineMarkdown(line.trim())}</p>`);
  }
  closeList();
  if (inCode) out.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
  return out.join('');
}

function formatDuration(ms) {
  const value = Number(ms || 0);
  if (value < 1000) return `${value} ms`;
  const seconds = value / 1000;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

function modeName(mode) {
  return ({auto:'Auto', direct:'Quick Answer', qa:'Quick Answer', research:'Comprehensive', deep:'Deep Analysis'})[mode] || mode;
}

function modeHint(mode) {
  return ({
    auto:'Best balance of speed and depth.',
    direct:'Fast focused Q&A with bounded retrieval.',
    research:'Broad evidence search, coverage checks and verification.',
    deep:'Long-form analysis across the chosen PDFs.'
  })[mode] || '';
}

function stageName(stage, fallback = '') {
  return ({
    queue:'Waiting for capacity', prepare:'Understanding question', route:'Choosing approach', retrieve:'Searching', retrieve_direct:'Searching', retrieve_expanded:'Searching wider',
    rerank:'Reranking', coverage:'Checking coverage', expand:'Mapping search', plan:'Planning response', answer:'Writing answer', verify:'Verifying', repair:'Applying corrections',
    queued:'Queued', analyze:'Analysing sections', reduce:'Combining findings', synthesize:'Writing analysis', complete:'Complete'
  })[stage] || fallback || 'Working';
}

function citationPageLabel(citation) {
  return citation.page_from == null
    ? 'page ?'
    : `page ${citation.page_from}${citation.page_to && citation.page_to !== citation.page_from ? '-' + citation.page_to : ''}`;
}

function citationHtml(citation) {
  const pages = citationPageLabel(citation);
  const section = citation.section_path?.length ? ` · ${escapeHtml(citation.section_path.join(' / '))}` : '';
  return `<div class="source"><div class="source-head"><strong>[${escapeHtml(citation.evidence_id)}] ${escapeHtml(citation.document_title)}</strong><span>${pages}</span></div><small>${escapeHtml(citation.filename)}${section}</small><div class="source-excerpt">${escapeHtml(citation.excerpt || '')}</div></div>`;
}

function ensureSourcePreview() {
  let preview = document.getElementById('sourceCitationPreview');
  if (preview) return preview;
  preview = document.createElement('aside');
  preview.id = 'sourceCitationPreview';
  preview.className = 'source-citation-preview';
  preview.setAttribute('role', 'tooltip');
  preview.setAttribute('aria-hidden', 'true');
  document.body.appendChild(preview);
  return preview;
}

function sourcePreviewHtml(citation) {
  const section = citation.section_path?.length
    ? `<div class="source-preview-section">${escapeHtml(citation.section_path.join(' / '))}</div>`
    : '';
  return `<div class="source-preview-head"><strong>[${escapeHtml(citation.evidence_id)}] ${escapeHtml(citation.document_title || citation.filename || 'Source')}</strong><span>${escapeHtml(citationPageLabel(citation))}</span></div><div class="source-preview-file">${escapeHtml(citation.filename || '')}</div>${section}<div class="source-preview-excerpt">${escapeHtml(citation.excerpt || 'No source excerpt is available for this citation.')}</div>`;
}

function positionSourcePreview(anchor, preview) {
  const rect = anchor.getBoundingClientRect();
  const margin = 10;
  const width = Math.min(430, Math.max(260, window.innerWidth - margin * 2));
  preview.style.width = `${width}px`;
  preview.style.left = `${Math.max(margin, Math.min(rect.left, window.innerWidth - width - margin))}px`;
  const measuredHeight = Math.min(preview.scrollHeight || 240, Math.max(160, window.innerHeight * 0.58));
  const below = rect.bottom + 8;
  const top = below + measuredHeight <= window.innerHeight - margin
    ? below
    : Math.max(margin, rect.top - measuredHeight - 8);
  preview.style.top = `${top}px`;
}

function showSourcePreview(anchor, citation, autoHide = false) {
  if (!citation) return;
  if (sourcePreviewTimer) { clearTimeout(sourcePreviewTimer); sourcePreviewTimer = null; }
  const preview = ensureSourcePreview();
  preview.innerHTML = sourcePreviewHtml(citation);
  preview.classList.add('visible');
  preview.setAttribute('aria-hidden', 'false');
  requestAnimationFrame(() => positionSourcePreview(anchor, preview));
  if (autoHide) sourcePreviewTimer = setTimeout(hideSourcePreview, 4500);
}

function hideSourcePreview() {
  if (sourcePreviewTimer) { clearTimeout(sourcePreviewTimer); sourcePreviewTimer = null; }
  const preview = document.getElementById('sourceCitationPreview');
  if (!preview) return;
  preview.classList.remove('visible');
  preview.setAttribute('aria-hidden', 'true');
}

function bindCitationPreviews(article, citations) {
  const sourceById = new Map((citations || []).map(citation => [String(citation.evidence_id || ''), citation]));
  article.querySelectorAll('.inline-citation[data-evidence-id]').forEach(anchor => {
    const citation = sourceById.get(String(anchor.dataset.evidenceId || ''));
    if (!citation) return;
    anchor.classList.add('source-preview-enabled');
    anchor.addEventListener('mouseenter', () => showSourcePreview(anchor, citation));
    anchor.addEventListener('mouseleave', hideSourcePreview);
    anchor.addEventListener('focus', () => showSourcePreview(anchor, citation));
    anchor.addEventListener('blur', hideSourcePreview);
    anchor.addEventListener('click', event => {
      event.preventDefault();
      showSourcePreview(anchor, citation, true);
    });
  });
}

async function copyTextToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const area = document.createElement('textarea');
  area.value = text;
  area.setAttribute('readonly', '');
  area.style.position = 'fixed';
  area.style.left = '-9999px';
  document.body.appendChild(area);
  area.select();
  const copied = document.execCommand('copy');
  area.remove();
  if (!copied) throw new Error('Clipboard copy was rejected by the browser.');
}

function visualEvidenceHtml(visuals) {
  if (!Array.isArray(visuals) || !visuals.length) return '';
  const cards = visuals.map(item => {
    const url = `/api/v1/visuals/render/${encodeURIComponent(item.token)}`;
    const evidence = item.evidence_id ? ` · [${escapeHtml(item.evidence_id)}]` : '';
    return `<figure class="visual-card"><a href="${url}" target="_blank" rel="noopener"><img loading="lazy" src="${url}" alt="${escapeHtml(item.kind || 'source visual')} from ${escapeHtml(item.document_title)}, page ${escapeHtml(item.page)}"></a><figcaption><strong>${escapeHtml(item.document_title)}</strong> · page ${escapeHtml(item.page)}${evidence}<br>${escapeHtml(item.caption || 'Relevant source visual')}</figcaption></figure>`;
  }).join('');
  return `<section class="visual-evidence"><h3>Related source visuals</h3><div class="visual-grid">${cards}</div></section>`;
}

function conversationScroller() {
  return $('conversationScroll');
}

function scrollConversationToBottom(behavior = 'smooth') {
  const scroller = conversationScroller();
  if (!scroller) return;
  requestAnimationFrame(() => scroller.scrollTo({top:scroller.scrollHeight, behavior}));
}

function scrollConversationToNode(node, behavior = 'smooth') {
  const scroller = conversationScroller();
  if (!scroller || !node) return;
  requestAnimationFrame(() => {
    const scrollerRect = scroller.getBoundingClientRect();
    const nodeRect = node.getBoundingClientRect();
    const top = scroller.scrollTop + nodeRect.top - scrollerRect.top - 10;
    scroller.scrollTo({top:Math.max(0, top), behavior});
  });
}

function createQueryCard(question, mode) {
  const article = document.createElement('article');
  article.className = 'query-card card';
  article.innerHTML = `<div class="question-strip"><small>${escapeHtml(modeName(mode))}</small><strong>${escapeHtml(question)}</strong></div><div class="answer-area"></div>`;
  return article;
}

function createLiveCard(question, mode) {
  const article = createQueryCard(question, mode);
  const area = article.querySelector('.answer-area');
  area.innerHTML = `<div class="working-row"><span class="spinner"></span><strong>Working</strong></div><div class="progress-track"><span></span></div><div class="progress-list"></div>`;
  const container = $('activeQuery');
  container.querySelector('.empty-answer')?.remove();
  container.appendChild(article);
  scrollConversationToBottom();
  return article;
}

function createRealtimeLiveCard(question) {
  const article = createQueryCard(question, 'direct');
  article.classList.add('realtime-query-card');
  const area = article.querySelector('.answer-area');
  area.innerHTML = `<div class="working-row"><span class="spinner"></span><strong>Finding applicable action</strong></div><div class="progress-track"><span></span></div><div class="progress-list"></div>`;
  const container = $('rtActiveQuery');
  container.querySelector('.empty-answer')?.remove();
  container.appendChild(article);
  requestAnimationFrame(() => $('rtConversationScroll')?.scrollTo({top:$('rtConversationScroll').scrollHeight, behavior:'smooth'}));
  return article;
}

function appendProgress(article, stage, label, percent) {
  const list = article.querySelector('.progress-list');
  const bar = article.querySelector('.progress-track span');
  if (!list || !bar) return;
  const existing = list.querySelector(`[data-stage="${CSS.escape(stage || label)}"]`);
  if (existing) {
    existing.classList.add('done');
    existing.querySelector('.progress-dot').textContent = '✓';
    existing.querySelector('span:last-child').textContent = label;
  } else {
    list.querySelectorAll('.progress-step:not(.done)').forEach(row => { row.classList.add('done'); row.querySelector('.progress-dot').textContent = '✓'; });
    const row = document.createElement('div');
    row.className = 'progress-step';
    row.dataset.stage = stage || label;
    row.innerHTML = `<span class="progress-dot">•</span><span>${escapeHtml(label)}</span>`;
    list.appendChild(row);
  }
  bar.style.width = `${Math.max(2, Math.min(100, Number(percent || 0)))}%`;
  scrollConversationToBottom();
}

function finishProgress(article) {
  article.querySelectorAll('.progress-step').forEach(row => { row.classList.add('done'); row.querySelector('.progress-dot').textContent = '✓'; });
  const bar = article.querySelector('.progress-track span');
  if (bar) bar.style.width = '100%';
}

function answerToolbar(mode, confidence, latency, createdAt) {
  const items = [`<span class="badge primary-badge">${escapeHtml(modeName(mode))}</span>`];
  if (confidence) items.push(`<span class="badge">${escapeHtml(confidence)} confidence</span>`);
  if (latency != null) items.push(`<span class="badge">${formatDuration(latency)}</span>`);
  const date = createdAt ? `<span class="subtle">${new Date(createdAt).toLocaleString()}</span>` : '';
  return `<div class="answer-toolbar"><div class="badges">${items.join('')}</div><div class="answer-actions">${date}<button class="copy-answer secondary" type="button" title="Copy answer" aria-label="Copy answer">Copy</button></div></div>`;
}

function renderFinalAnswer(article, result, createdAt = null, keepProgress = true) {
  const area = article.querySelector('.answer-area');
  const workflow = keepProgress ? area.querySelector('.progress-list')?.outerHTML || '' : '';
  const workflowDetails = workflow ? `<details class="workflow-details"><summary>Processing</summary>${workflow}</details>` : '';
  const citations = result.citations?.length ? result.citations.map(citationHtml).join('') : '<p class="warning">No source citations returned.</p>';
  answerTextByCard.set(article, result.answer || '');
  area.innerHTML = `${answerToolbar(result.resolved_mode || result.mode, result.confidence, result.latency_ms, createdAt)}${workflowDetails}<div class="markdown-body">${renderMarkdown(result.answer)}</div>${visualEvidenceHtml(result.visuals)}<details class="source-details"><summary>Sources · ${result.citations?.length || 0}</summary><div class="sources">${citations}</div></details>${result.query_id ? `<div class="feedback-row"><span>Helpful?</span><button class="feedback-button" data-query-id="${escapeHtml(result.query_id)}" data-rating="5">Yes</button><button class="feedback-button" data-query-id="${escapeHtml(result.query_id)}" data-rating="2">Needs work</button></div>` : ''}`;
  bindCitationPreviews(article, result.citations || []);
  if (article.closest('#activeQuery')) scrollConversationToNode(area.querySelector('.answer-toolbar') || area);
}

function renderDeepAnswer(article, report, createdAt = null) {
  const area = article.querySelector('.answer-area');
  const citations = report.citations?.length ? report.citations.map(citationHtml).join('') : '<p class="warning">No source citations returned.</p>';
  answerTextByCard.set(article, report.result_markdown || '');
  area.innerHTML = `${answerToolbar('deep', null, null, createdAt)}<div class="markdown-body">${renderMarkdown(report.result_markdown || '')}</div><details class="source-details"><summary>Sources · ${report.citations?.length || 0}</summary><div class="sources">${citations}</div></details>`;
  bindCitationPreviews(article, report.citations || []);
  if (article.closest('#activeQuery')) scrollConversationToNode(area.querySelector('.answer-toolbar') || area);
}

async function streamQuery(payload, onEvent, signal = null) {
  const response = await fetch('/api/v1/query/stream', {method:'POST', credentials:'same-origin', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload), signal});
  if (!response.ok) {
    const raw = await response.text();
    let data = null;
    if (raw) { try { data = JSON.parse(raw); } catch { data = {detail:raw}; } }
    throw new Error(validationMessage(data?.detail || data || `Request failed (${response.status})`));
  }
  if (!response.body) throw new Error('Streaming responses are not supported by this browser.');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const {value, done} = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), {stream:!done});
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    for (const line of lines) if (line.trim()) onEvent(JSON.parse(line));
    if (done) break;
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer));
}

function scopeIds(forDeep = false) {
  const scope = $('queryScope').value;
  if (scope === 'selected') return [...selectedQueryDocumentIds];
  if (scope === 'knowledge') return orgDocs.filter(doc => doc.ingestion_status === 'ready').map(doc => doc.id);
  if (scope === 'personal') return personalDocs.filter(doc => doc.ingestion_status === 'ready').map(doc => doc.id);
  return forDeep ? readyDocs().map(doc => doc.id) : null;
}

function resizeQuestionInput() {
  const input = $('question');
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
}

function setSourcePicker(open) {
  const panel = $('selectedSourcePanel');
  const button = $('choosePdfsBtn');
  panel.classList.toggle('hidden', !open);
  button.setAttribute('aria-expanded', String(open));
  button.classList.toggle('active', open || $('queryScope').value === 'selected');
  if (open) {
    renderDocumentPicker();
    setTimeout(() => $('queryDocumentSearch').focus(), 0);
  }
}

function updateSelectedDocumentState() {
  const validIds = new Set(readyDocs().map(doc => String(doc.id)));
  selectedQueryDocumentIds = new Set([...selectedQueryDocumentIds].map(String).filter(id => validIds.has(id)));
  const count = selectedQueryDocumentIds.size;
  $('selectedDocCount').textContent = String(count);
  $('sourcePickerSummary').textContent = `${count} selected`;
  $('choosePdfsBtn').classList.toggle('active', $('queryScope').value === 'selected' || !$('selectedSourcePanel').classList.contains('hidden'));
}

function setAskRunning(running) {
  const button = $('askBtn');
  button.classList.toggle('stop-button', running);
  $('askBtnLabel').textContent = running ? 'Stop' : (selectedMode === 'deep' ? 'Analyse' : 'Ask');
  button.disabled = false;
}

async function stopActiveQuery() {
  const active = activeQaRequest;
  if (!active) return;
  try {
    await fetch(`/api/v1/query/${encodeURIComponent(active.requestId)}/cancel`, {
      method:'POST', credentials:'same-origin'
    });
  } catch (_) { /* best effort; AbortController still closes the client stream */ }
  active.controller.abort();
  $('question').value = active.question;
  resizeQuestionInput();
  const area = active.live.querySelector('.answer-area');
  area.insertAdjacentHTML('beforeend', '<div class="status-message">Stopped. Edit the question below and ask again when ready.</div>');
  activeQaRequest = null;
  setAskRunning(false);
  $('question').disabled = false;
  $('question').focus();
}

async function askQuestion() {
  if (activeQaRequest) { await stopActiveQuery(); return; }
  const question = $('question').value.trim();
  if (!question) { $('queryStatus').textContent = 'Enter a question.'; return; }
  let ids = scopeIds(selectedMode === 'deep');
  if ($('queryScope').value !== 'all' && (!ids || !ids.length)) { $('queryStatus').textContent = 'No ready PDFs are selected for this source scope.'; return; }
  if (selectedMode === 'deep') {
    ids = ids || [];
    if (!ids.length) { $('queryStatus').textContent = 'Deep Analysis needs at least one ready PDF.'; return; }
    if (ids.length > reportMaxDocuments) { $('queryStatus').textContent = `Deep Analysis supports up to ${reportMaxDocuments} PDFs per run. Choose a smaller source set.`; return; }
  }

  $('queryStatus').textContent = '';
  $('question').value = '';
  resizeQuestionInput();
  setSourcePicker(false);
  const live = createLiveCard(question, selectedMode);
  setAskRunning(true);
  try {
    if (selectedMode === 'deep') {
      $('askBtn').disabled = true;
      await startDeepAnalysis(question, ids, live);
    } else {
      const requestId = crypto.randomUUID();
      const controller = new AbortController();
      activeQaRequest = {requestId, controller, question, live};
      await startQa(question, ids, live, requestId, controller);
    }
    await Promise.all([loadHistory(), loadDashboard()]);
  } catch (error) {
    if (error?.name !== 'AbortError') {
      live.querySelector('.answer-area').innerHTML += `<div class="status-message error"><strong>Could not complete request.</strong> ${escapeHtml(error.message)}</div>`;
      scrollConversationToBottom();
    }
  } finally {
    activeQaRequest = null;
    $('askBtn').disabled = false;
    setAskRunning(false);
    $('question').disabled = false;
    $('question').focus();
  }
}

async function startQa(question, ids, live, requestId, controller) {
  let finished = false;
  let cancelled = false;
  let streamingBody = null;
  await streamQuery({request_id:requestId, question, mode:selectedMode, document_ids:ids}, event => {
    if (event.type === 'heartbeat') return;
    if (event.type === 'progress') appendProgress(live, event.stage, stageName(event.stage, event.label), event.percent);
    if (event.type === 'answer_delta') {
      if (!streamingBody) {
        streamingBody = document.createElement('div');
        streamingBody.className = 'markdown-body streaming-answer';
        live.querySelector('.answer-area').appendChild(streamingBody);
      }
      streamingBody.textContent += event.delta || '';
      scrollConversationToBottom('auto');
    }
    if (event.type === 'cancelled') { cancelled = true; }
    if (event.type === 'error') throw new Error(event.message || 'Query failed');
    if (event.type === 'result') { finished = true; finishProgress(live); renderFinalAnswer(live, event.data, null, true); }
  }, controller.signal);
  if (cancelled) return;
  if (!finished) throw new Error('The query ended before an answer was returned.');
}

async function startDeepAnalysis(question, ids, live) {
  appendProgress(live, 'queued', 'Queuing analysis', 3);
  const report = await api('/api/v1/reports', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({title:question.slice(0, 120), objective:question, document_ids:ids})});
  let current = report;
  while (['queued','running'].includes(current.status)) {
    appendProgress(live, current.progress_stage || current.status, stageName(current.progress_stage || current.status, current.progress_message), current.progress_percent || 5);
    await new Promise(resolve => setTimeout(resolve, 2500));
    current = await api(`/api/v1/reports/${report.id}`);
  }
  if (current.status === 'failed') throw new Error(current.error || 'Deep Analysis failed.');
  finishProgress(live);
  renderDeepAnswer(live, current);
}

function collectRealtimeContext() {
  return {
    line:$('rtLine').value || null, rolling_stock:$('rtRollingStock').value || null,
    role:$('rtRole').value || null, department:$('rtDepartment').value || null,
    operating_mode:$('rtMode').value || null, location_type:$('rtLocationType').value || null,
    station_or_section:$('rtStation').value.trim() || null
  };
}

function renderRealtimeContext() {
  const values = [realtimeContext?.line, realtimeContext?.rolling_stock, realtimeContext?.role, realtimeContext?.department, realtimeContext?.operating_mode, realtimeContext?.location_type].filter(Boolean);
  $('rtContextSummary').textContent = values.length ? values.join(' · ') : 'No session filters';
}

function startRealtimeSession() {
  const context = collectRealtimeContext();
  realtimeContext = context;
  $('rtSetupStatus').textContent = '';
  $('realtimeSetup').classList.add('hidden');
  $('realtimeSession').classList.remove('hidden');
  renderRealtimeContext();
  $('rtQuestion').focus();

  // Persistence is optional. A browser storage failure must never block an operational session.
  try { sessionStorage.setItem('ims-realtime-context', JSON.stringify(context)); } catch (_) {}
}

function changeRealtimeContext() {
  $('realtimeSession').classList.add('hidden');
  $('realtimeSetup').classList.remove('hidden');
}

async function askRealtime() {
  if (activeRealtimeRequest) return;
  const question = $('rtQuestion').value.trim();
  if (!question) { $('rtStatus').textContent = 'Describe the condition or ask what to do.'; return; }
  if (!realtimeContext) { $('rtStatus').textContent = 'Set operational context first.'; return; }
  $('rtStatus').textContent = ''; $('rtQuestion').value = '';
  const live = createRealtimeLiveCard(question);
  const requestId = crypto.randomUUID(); const controller = new AbortController();
  activeRealtimeRequest = {requestId, controller}; $('rtAskBtn').disabled = true; $('rtQuestion').disabled = true;
  try {
    let finished = false;
    await streamQuery({request_id:requestId, question, mode:'direct', experience:'realtime', operational_context:realtimeContext, document_ids:null}, event => {
      if (event.type === 'heartbeat') return;
      if (event.type === 'progress') appendProgress(live, event.stage, stageName(event.stage, event.label), event.percent);
      if (event.type === 'error') throw new Error(event.message || 'Query failed');
      if (event.type === 'result') { finished = true; finishProgress(live); renderFinalAnswer(live, event.data, null, true); }
    }, controller.signal);
    if (!finished) throw new Error('The real-time query ended before an answer was returned.');
  } catch (error) {
    live.querySelector('.answer-area').innerHTML += `<div class="status-message error"><strong>Could not complete request.</strong> ${escapeHtml(error.message)}</div>`;
  } finally { activeRealtimeRequest = null; $('rtAskBtn').disabled = false; $('rtQuestion').disabled = false; $('rtQuestion').focus(); }
}

function setMode(mode) {
  selectedMode = mode;
  $('queryMode').value = mode;
  $('modeHint').textContent = modeHint(mode);
  $('askBtnLabel').textContent = mode === 'deep' ? 'Analyse' : 'Ask';
}

function documentSearchText(doc) {
  return [doc.title, doc.original_filename, doc.family_key, doc.revision, doc.authority, doc.department, doc.ingestion_status]
    .filter(Boolean).join(' ').toLowerCase();
}

function renderDocumentPicker() {
  updateSelectedDocumentState();
  const term = $('queryDocumentSearch').value.trim().toLowerCase();
  const groups = [
    ['Knowledge', orgDocs.filter(doc => doc.ingestion_status === 'ready' && (!term || documentSearchText(doc).includes(term)))],
    ['My PDFs', personalDocs.filter(doc => doc.ingestion_status === 'ready' && (!term || documentSearchText(doc).includes(term)))],
  ].filter(([, values]) => values.length);
  $('queryDocuments').innerHTML = groups.length
    ? groups.map(([label, values]) => `<div class="picker-group">${label} · ${values.length}</div>${values.map(doc => `<label><input type="checkbox" value="${doc.id}" ${selectedQueryDocumentIds.has(String(doc.id)) ? 'checked' : ''}><span>${escapeHtml(doc.title)}<small>${escapeHtml(doc.original_filename || '')}</small></span></label>`).join('')}`).join('')
    : '<div class="picker-empty">No ready PDFs match this search.</div>';
}

function docStatusStats(values) {
  const count = status => values.filter(doc => doc.ingestion_status === status).length;
  return `<div><strong>${values.length}</strong><span>Total</span></div><div><strong>${count('ready')}</strong><span>Ready</span></div><div><strong>${count('processing') + count('queued')}</strong><span>Indexing</span></div><div><strong>${count('failed')}</strong><span>Failed</span></div>`;
}

function documentRow(doc, personal = false) {
  const pages = doc.page_count ? `${doc.page_count} pages` : '';
  const meta = [doc.original_filename, pages].filter(Boolean).map(escapeHtml).join(' · ');
  const error = doc.ingestion_error ? `<small class="error">${escapeHtml(doc.ingestion_error)}</small>` : '';
  const shareChips = personal && doc.shared_with?.length ? `<div class="share-chips">${doc.shared_with.map(user => `<span class="share-chip">${escapeHtml(user.username)}</span>`).join('')}</div>` : '';
  const ownerLabel = personal ? `<small>${doc.is_owner ? 'Owned by you' : 'Shared with you'}</small>` : '';
  const retry = doc.can_manage && doc.ingestion_status === 'failed' ? `<button data-doc-action="retry" data-doc-id="${doc.id}">Retry</button>` : '';
  const reindex = doc.can_manage && doc.ingestion_status === 'ready' ? `<button data-doc-action="reindex" data-doc-id="${doc.id}">Reprocess</button>` : '';
  const share = doc.can_share ? `<button data-share-doc="${doc.id}">Manage access</button>` : '';
  const del = doc.can_manage && ['ready','failed'].includes(doc.ingestion_status) ? `<button class="danger-action" data-doc-action="delete" data-doc-id="${doc.id}" data-doc-title="${escapeHtml(doc.title)}">Delete</button>` : '';
  return `<div class="doc-row"><div class="doc-top"><div class="doc-main"><strong>${escapeHtml(doc.title)}</strong><small>${meta}</small>${ownerLabel}${shareChips}${error}</div><span class="status-pill ${escapeHtml(doc.ingestion_status)}">${escapeHtml(doc.ingestion_status)}</span></div><div class="doc-actions"><a href="/api/v1/documents/${doc.id}/view" target="_blank" rel="noopener">View</a><a href="/api/v1/documents/${doc.id}/download">Download</a>${retry}${reindex}${share}${del}</div></div>`;
}

function statusMatches(doc, filter) {
  if (filter === 'all') return true;
  if (filter === 'indexing') return ['queued','processing'].includes(doc.ingestion_status);
  return doc.ingestion_status === filter;
}

function renderDocumentList(values, {searchId, statusId, listId, metaId, personal}) {
  const term = $(searchId).value.trim().toLowerCase();
  const status = $(statusId).value;
  const matches = values.filter(doc => (!term || documentSearchText(doc).includes(term)) && statusMatches(doc, status));
  $(metaId).textContent = `Showing ${matches.length} of ${values.length} PDFs`;
  $(listId).innerHTML = matches.length
    ? matches.map(doc => documentRow(doc, personal)).join('')
    : '<p class="subtle">No PDFs match the current search and status filter.</p>';
}

function renderFilteredDocuments() {
  renderDocumentList(orgDocs, {searchId:'documentSearch', statusId:'documentStatusFilter', listId:'documentList', metaId:'documentListMeta', personal:false});
  renderDocumentList(personalDocs, {searchId:'personalDocumentSearch', statusId:'personalStatusFilter', listId:'personalDocumentList', metaId:'personalListMeta', personal:true});
}

function renderDocuments() {
  $('documentStats').innerHTML = docStatusStats(orgDocs);
  $('personalStats').innerHTML = docStatusStats(personalDocs);
  renderFilteredDocuments();
  renderDocumentPicker();
  renderInsightRail();
  scheduleDocumentPoll();
}

function scheduleDocumentPoll() {
  if (documentPollTimer) clearTimeout(documentPollTimer);
  if (allDocs().some(doc => ['queued','processing'].includes(doc.ingestion_status))) {
    documentPollTimer = setTimeout(() => loadDocs().catch(() => {}), 5000);
  }
}

async function loadDocs() {
  [orgDocs, personalDocs] = await Promise.all([api('/api/v1/documents'), api('/api/v1/library')]);
  renderDocuments();
}

function renderDirectoryPickers() {
  const html = directoryUsers.length ? directoryUsers.map(user => `<label><input type="checkbox" value="${user.id}"><span>${escapeHtml(user.username)}${user.department ? ` · ${escapeHtml(user.department)}` : ''}</span></label>`).join('') : '<span class="subtle">No other active users.</span>';
  $('personalUploadShares').innerHTML = html;
}

async function loadDirectory() {
  directoryUsers = await api('/api/v1/auth/directory');
  renderDirectoryPickers();
}

async function uploadFiles({files, endpoint, statusNode, title, sharedUserIds = []}) {
  let done = 0;
  let failed = 0;
  const failures = [];
  statusNode.className = 'status-message';
  statusNode.textContent = `Uploading 0/${files.length}...`;
  async function one(file) {
    const payload = new FormData();
    payload.append('file', file, file.name);
    if (title && files.length === 1) payload.append('title', title);
    if (sharedUserIds.length) payload.append('shared_user_ids', sharedUserIds.join(','));
    try { await api(endpoint, {method:'POST', body:payload}); }
    catch (error) { failed++; failures.push(`${file.name}: ${error.message}`); }
    finally { done++; statusNode.textContent = `Uploaded ${done}/${files.length}${failed ? ` · ${failed} failed` : ''}`; }
  }
  for (let start = 0; start < files.length; start += 3) await Promise.all(files.slice(start,start+3).map(one));
  if (failed) { statusNode.className = 'status-message error'; statusNode.textContent = `${files.length - failed} accepted, ${failed} failed. ${failures.slice(0,3).join(' · ')}`; }
  else { statusNode.className = 'status-message success'; statusNode.textContent = `${files.length} PDF${files.length === 1 ? '' : 's'} queued.`; }
  return failed;
}

function normalizeHistory(questions, reports) {
  const q = questions.map(item => ({type:'qa', created_at:item.created_at, item}));
  const r = reports.map(item => ({type:'deep', created_at:item.created_at, item}));
  return [...q, ...r].sort((a,b) => new Date(b.created_at) - new Date(a.created_at));
}

function renderHistory() {
  const term = $('historySearch').value.trim().toLowerCase();
  const matches = historyItems.filter(entry => {
    const item = entry.item;
    const haystack = entry.type === 'qa' ? `${item.question} ${item.answer}` : `${item.objective} ${item.result_markdown || ''}`;
    return !term || haystack.toLowerCase().includes(term);
  });
  const container = $('questionHistory');
  container.innerHTML = '';
  if (!matches.length) { container.innerHTML = '<div class="card history-empty subtle">No matching items.</div>'; return; }
  matches.forEach(entry => {
    const item = entry.item;
    if (entry.type === 'qa') {
      const card = createQueryCard(item.question, item.mode);
      renderFinalAnswer(card, item, item.created_at, false);
      container.appendChild(card);
    } else {
      const card = createQueryCard(item.objective, 'deep');
      if (item.status === 'complete') renderDeepAnswer(card, item, item.created_at);
      else {
        const area = card.querySelector('.answer-area');
        area.innerHTML = `${answerToolbar('deep', null, null, item.created_at)}<div class="progress-track"><span style="width:${Number(item.progress_percent || 0)}%"></span></div><p class="subtle">${escapeHtml(item.progress_message || item.status)}</p>${item.error ? `<p class="error">${escapeHtml(item.error)}</p>` : ''}`;
      }
      container.appendChild(card);
    }
  });
}

async function loadHistory() {
  const [questions, reports] = await Promise.all([api('/api/v1/query/history?limit=100'), api('/api/v1/reports')]);
  historyItems = normalizeHistory(questions, reports);
  renderHistory();
  renderInsightRail();
}

function renderInsightRail() {
  if (!me) return;
  const initials = String(me.username || 'U').trim().slice(0, 2).toUpperCase();
  $('profileAvatar').textContent = initials || 'U';
  $('profileUsername').textContent = me.username || 'User';
  $('profileRole').textContent = me.role || 'user';
  $('profileDepartment').textContent = me.department || 'No department';

  const questionCount = dashboardSummary?.question_count ?? historyItems.filter(entry => entry.type === 'qa').length;
  const availablePdfCount = allDocs().length;
  const readyPdfCount = readyDocs().length;
  const indexingPdfCount = allDocs().filter(doc => ['queued','processing'].includes(doc.ingestion_status)).length;
  $('dashboardStats').innerHTML = `<div><strong>${questionCount}</strong><span>Questions</span></div><div><strong>${availablePdfCount}</strong><span>PDFs available</span></div><div><strong>${readyPdfCount}</strong><span>Ready</span></div><div><strong>${indexingPdfCount}</strong><span>Indexing</span></div>`;

  let recent = dashboardSummary?.recent_questions || [];
  if (!recent.length) {
    recent = historyItems.filter(entry => entry.type === 'qa').slice(0, 5).map(entry => ({
      question:entry.item.question,
      resolved_mode:entry.item.resolved_mode || entry.item.mode,
      created_at:entry.item.created_at,
    }));
  }
  $('recentQuestionList').innerHTML = recent.length
    ? recent.map(item => `<button class="recent-question-button" type="button" data-recent-question="${escapeHtml(item.question)}"><strong>${escapeHtml(item.question)}</strong><span>${escapeHtml(modeName(item.resolved_mode || item.mode))} · ${new Date(item.created_at).toLocaleDateString()}</span></button>`).join('')
    : '<p class="subtle">No recent questions yet.</p>';
}

async function loadDashboard() {
  try { dashboardSummary = await api('/api/v1/dashboard/summary'); }
  catch { dashboardSummary = null; }
  renderInsightRail();
}

async function loadUsers() {
  if (me?.role !== 'admin') return;
  const users = await api('/api/v1/auth/users');
  $('userList').innerHTML = users.map(user => `<div class="user-row"><div><strong>${escapeHtml(user.username)}</strong><small>${escapeHtml(user.role)}${user.department ? ` · ${escapeHtml(user.department)}` : ''} · ${user.is_active ? 'active' : 'inactive'}</small></div><div class="user-actions"><button data-user-action="password" data-user-id="${user.id}" data-user-name="${escapeHtml(user.username)}">Reset password</button>${String(user.id) !== String(me.id) ? `<button data-user-action="status" data-user-id="${user.id}" data-active="${user.is_active ? 'false' : 'true'}">${user.is_active ? 'Deactivate' : 'Activate'}</button>` : ''}</div></div>`).join('');
}


function setActivityDrawer(open) {
  const drawer = $('activityDrawer');
  const backdrop = $('activityBackdrop');
  if (!drawer || !backdrop) return;
  drawer.classList.toggle('open', open);
  drawer.setAttribute('aria-hidden', String(!open));
  backdrop.classList.toggle('hidden', !open);
  backdrop.setAttribute('aria-hidden', String(!open));
  document.body.classList.toggle('drawer-open', open);
  if (open) loadDashboard().catch(() => {});
}

function switchTab(name) {
  setActivityDrawer(false);
  document.querySelector('.sidebar-query-controls')?.classList.toggle('hidden', name === 'realtime');
  document.querySelectorAll('[data-tab]').forEach(button => button.classList.toggle('active', button.dataset.tab === name));
  document.querySelectorAll('.tab-content').forEach(content => content.classList.add('hidden'));
  $(`tab-${name}`).classList.remove('hidden');
  if (name === 'history') loadHistory().catch(() => {});
  if (name === 'documents' || name === 'library') loadDocs().catch(() => {});
  if (name === 'library') loadDirectory().catch(() => {});
  if (name === 'users') loadUsers().catch(() => {});
  window.scrollTo({top:0, behavior:'smooth'});
}

function applyTheme(theme) {
  document.body.dataset.theme = theme;
  $('themeToggle').textContent = theme === 'dark' ? '☀' : '☾';
  $('themeToggle').title = theme === 'dark' ? 'Use light theme' : 'Use dark theme';
  localStorage.setItem('ike-theme', theme);
}

function initialTheme() {
  return localStorage.getItem('ike-theme') || (window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
}

function showApp() {
  $('loginPanel').classList.add('hidden');
  $('appPanel').classList.remove('hidden');
  $('userBox').innerHTML = `<div class="user-meta"><strong>${escapeHtml(me.username)}</strong><small>${escapeHtml(me.role)}${me.department ? ` · ${escapeHtml(me.department)}` : ''}</small></div><button id="logoutBtn" class="logout-button">Sign out</button>`;
  $('usersTabButton').classList.toggle('hidden', me.role !== 'admin');
  const canUploadKnowledge = ['admin','analyst'].includes(me.role);
  $('uploadCard').classList.toggle('hidden', !canUploadKnowledge);
  if (!canUploadKnowledge) $('uploadCard').parentElement.style.gridTemplateColumns = '1fr';
  $('logoutBtn').addEventListener('click', async () => { await api('/api/v1/auth/logout', {method:'POST'}); location.reload(); });
  renderInsightRail();
  if ($('rtDepartment')) $('rtDepartment').value = me.department || '';
  try {
    const saved = JSON.parse(sessionStorage.getItem('ims-realtime-context') || 'null');
    if (saved) { realtimeContext = saved; }
  } catch (_) { realtimeContext = null; }
}

async function boot() {
  applyTheme(initialTheme());
  setMode('research');
  resizeQuestionInput();
  try {
    me = await api('/api/v1/auth/me');
    showApp();
    if (realtimeContext) {
      if (realtimeContext.line) $('rtLine').value = realtimeContext.line;
      if (realtimeContext.rolling_stock) $('rtRollingStock').value = realtimeContext.rolling_stock;
      if (realtimeContext.role) $('rtRole').value = realtimeContext.role;
      if (realtimeContext.operating_mode) $('rtMode').value = realtimeContext.operating_mode;
      if (realtimeContext.location_type) $('rtLocationType').value = realtimeContext.location_type;
      if (realtimeContext.station_or_section) $('rtStation').value = realtimeContext.station_or_section;
    }
    await Promise.all([loadDocs(), loadHistory(), loadDirectory(), loadDashboard()]);
  } catch {
    $('loginPanel').classList.remove('hidden');
  }
}

$('themeToggle').addEventListener('click', () => applyTheme(document.body.dataset.theme === 'dark' ? 'light' : 'dark'));
$('activityToggle').addEventListener('click', () => setActivityDrawer(!$('activityDrawer').classList.contains('open')));
$('activitySidebarBtn').addEventListener('click', () => setActivityDrawer(true));
$('activityClose').addEventListener('click', () => setActivityDrawer(false));
$('activityBackdrop').addEventListener('click', () => setActivityDrawer(false));
$('modeInfoBtn').addEventListener('click', () => $('modeGuide').showModal());
document.querySelectorAll('[data-close-modal]').forEach(button => button.addEventListener('click', () => $(button.dataset.closeModal).close()));
document.querySelectorAll('[data-tab]').forEach(button => button.addEventListener('click', () => switchTab(button.dataset.tab)));
$('queryMode').addEventListener('change', event => setMode(event.target.value));
$('queryScope').addEventListener('change', event => {
  if (event.target.value === 'selected') setSourcePicker(true);
  else setSourcePicker(false);
  updateSelectedDocumentState();
});
$('choosePdfsBtn').addEventListener('click', () => {
  $('queryScope').value = 'selected';
  setSourcePicker($('selectedSourcePanel').classList.contains('hidden'));
  updateSelectedDocumentState();
});
$('queryDocumentSearch').addEventListener('input', renderDocumentPicker);
$('queryDocuments').addEventListener('change', event => {
  if (!event.target.matches('input[type="checkbox"]')) return;
  const id = String(event.target.value);
  if (event.target.checked) selectedQueryDocumentIds.add(id);
  else selectedQueryDocumentIds.delete(id);
  $('queryScope').value = 'selected';
  updateSelectedDocumentState();
});
$('clearSelectedDocs').addEventListener('click', () => { selectedQueryDocumentIds.clear(); renderDocumentPicker(); updateSelectedDocumentState(); });

document.addEventListener('click', async event => {
  const button = event.target.closest?.('.copy-answer');
  if (!button) return;
  const article = button.closest('.query-card');
  const text = article ? (answerTextByCard.get(article) || article.querySelector('.markdown-body')?.innerText || '') : '';
  try {
    await copyTextToClipboard(text);
    const previous = button.textContent; button.textContent = 'Copied';
    setTimeout(() => { button.textContent = previous || 'Copy'; }, 1400);
  } catch (_) {
    button.textContent = 'Copy failed';
    setTimeout(() => { button.textContent = 'Copy'; }, 1600);
  }
});
$('startRealtime').addEventListener('click', startRealtimeSession);
$('changeRealtimeContext').addEventListener('click', changeRealtimeContext);
$('rtAskBtn').addEventListener('click', askRealtime);
$('rtQuestion').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); askRealtime(); } });
$('askBtn').addEventListener('click', askQuestion);
$('question').addEventListener('input', resizeQuestionInput);
$('question').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); askQuestion(); }
});
$('refreshHistory').addEventListener('click', () => Promise.all([loadHistory(), loadDashboard()]).catch(() => {}));
$('historySearch').addEventListener('input', renderHistory);
$('refreshDocs').addEventListener('click', () => loadDocs().catch(() => {}));
$('refreshLibrary').addEventListener('click', () => Promise.all([loadDocs(), loadDirectory()]).catch(() => {}));
$('refreshUsers').addEventListener('click', () => loadUsers().catch(() => {}));
$('documentSearch').addEventListener('input', renderFilteredDocuments);
$('documentStatusFilter').addEventListener('change', renderFilteredDocuments);
$('personalDocumentSearch').addEventListener('input', renderFilteredDocuments);
$('personalStatusFilter').addEventListener('change', renderFilteredDocuments);
$('viewAllHistory').addEventListener('click', () => switchTab('history'));

$('loginForm').addEventListener('submit', async event => {
  event.preventDefault();
  $('loginError').textContent = '';
  try {
    await api('/api/v1/auth/login', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({username:$('username').value, password:$('password').value})});
    me = await api('/api/v1/auth/me');
    showApp();
    await Promise.all([loadDocs(), loadHistory(), loadDirectory(), loadDashboard()]);
  } catch (error) { $('loginError').textContent = error.message; }
});

$('uploadForm').addEventListener('submit', async event => {
  event.preventDefault();
  const files = [...$('docFile').files];
  if (!files.length) return;
  const form = new FormData(event.target);
  form.delete('file');
  const shared = [...form.entries()];
  let done = 0, failed = 0;
  const failures = [];
  $('uploadStatus').textContent = `Uploading 0/${files.length}...`;
  async function one(file) {
    const payload = new FormData();
    shared.forEach(([key,value]) => { if (key === 'title' && (files.length > 1 || !String(value).trim())) return; payload.append(key,value); });
    payload.append('file', file, file.name);
    try { await api('/api/v1/documents', {method:'POST', body:payload}); }
    catch (error) { failed++; failures.push(`${file.name}: ${error.message}`); }
    finally { done++; $('uploadStatus').textContent = `Uploaded ${done}/${files.length}${failed ? ` · ${failed} failed` : ''}`; }
  }
  for (let start = 0; start < files.length; start += 3) await Promise.all(files.slice(start,start+3).map(one));
  if (failed) { $('uploadStatus').className = 'status-message error'; $('uploadStatus').textContent = `${files.length-failed} accepted, ${failed} failed. ${failures.slice(0,3).join(' · ')}`; }
  else { $('uploadStatus').className = 'status-message success'; $('uploadStatus').textContent = `${files.length} PDF${files.length === 1 ? '' : 's'} queued.`; event.target.reset(); }
  await loadDocs();
});

$('personalUploadForm').addEventListener('submit', async event => {
  event.preventDefault();
  const files = [...$('personalDocFile').files];
  if (!files.length) return;
  const shareIds = [...document.querySelectorAll('#personalUploadShares input:checked')].map(input => input.value);
  const failed = await uploadFiles({files, endpoint:'/api/v1/library', statusNode:$('personalUploadStatus'), title:$('personalDocTitle').value.trim(), sharedUserIds:shareIds});
  if (!failed) event.target.reset();
  await loadDocs();
});

$('saveShares').addEventListener('click', async () => {
  if (!activeShareDocumentId) return;
  const ids = [...document.querySelectorAll('#shareUserPicker input:checked')].map(input => input.value);
  $('shareStatus').textContent = '';
  try {
    await api(`/api/v1/library/${activeShareDocumentId}/shares`, {method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify({user_ids:ids})});
    $('shareModal').close();
    activeShareDocumentId = null;
    await loadDocs();
  } catch (error) { $('shareStatus').className = 'status-message error'; $('shareStatus').textContent = error.message; }
});

$('userForm').addEventListener('submit', async event => {
  event.preventDefault();
  $('userStatus').textContent = '';
  try {
    await api('/api/v1/auth/users', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({username:$('newUsername').value.trim(), password:$('newUserPassword').value, role:$('newUserRole').value, department:$('newUserDepartment').value.trim() || null})});
    event.target.reset();
    $('userStatus').className = 'status-message success';
    $('userStatus').textContent = 'User created.';
    await Promise.all([loadUsers(), loadDirectory()]);
  } catch (error) { $('userStatus').className = 'status-message error'; $('userStatus').textContent = error.message; }
});

document.addEventListener('click', async event => {
  const recentQuestion = event.target.closest('[data-recent-question]');
  if (recentQuestion) {
    switchTab('history');
    $('historySearch').value = recentQuestion.dataset.recentQuestion || '';
    renderHistory();
    return;
  }

  const feedback = event.target.closest('.feedback-button');
  if (feedback) {
    try {
      await api('/api/v1/query/feedback', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({query_id:feedback.dataset.queryId, rating:Number(feedback.dataset.rating)})});
      feedback.closest('.feedback-row').innerHTML = '<span class="subtle">Feedback recorded.</span>';
    } catch (error) { feedback.closest('.feedback-row').innerHTML += `<span class="error">${escapeHtml(error.message)}</span>`; }
    return;
  }

  const docAction = event.target.closest('[data-doc-action]');
  if (docAction) {
    const id = docAction.dataset.docId;
    const action = docAction.dataset.docAction;
    try {
      if (action === 'delete') {
        if (!confirm(`Delete ${docAction.dataset.docTitle}? The PDF and its index will be permanently removed.`)) return;
        await api(`/api/v1/documents/${id}`, {method:'DELETE'});
      } else await api(`/api/v1/documents/${id}/${action}`, {method:'POST'});
      await loadDocs();
    } catch (error) {
      const node = $('tab-library').classList.contains('hidden') ? $('uploadStatus') : $('personalUploadStatus');
      node.className = 'status-message error'; node.textContent = error.message;
    }
    return;
  }

  const shareButton = event.target.closest('[data-share-doc]');
  if (shareButton) {
    const doc = personalDocs.find(item => item.id === shareButton.dataset.shareDoc);
    if (!doc) return;
    activeShareDocumentId = doc.id;
    $('shareModalTitle').textContent = doc.title;
    const selected = new Set((doc.shared_with || []).map(user => String(user.id)));
    $('shareUserPicker').innerHTML = directoryUsers.length ? directoryUsers.map(user => `<label><input type="checkbox" value="${user.id}" ${selected.has(String(user.id)) ? 'checked' : ''}><span>${escapeHtml(user.username)}${user.department ? ` · ${escapeHtml(user.department)}` : ''}</span></label>`).join('') : '<span class="subtle">No other active users.</span>';
    $('shareStatus').textContent = '';
    $('shareModal').showModal();
    return;
  }

  const userAction = event.target.closest('[data-user-action]');
  if (userAction) {
    try {
      if (userAction.dataset.userAction === 'status') {
        await api(`/api/v1/auth/users/${userAction.dataset.userId}/status`, {method:'PATCH', headers:{'Content-Type':'application/json'}, body:JSON.stringify({is_active:userAction.dataset.active === 'true'})});
      } else {
        const password = prompt(`New temporary password for ${userAction.dataset.userName} (minimum 12 characters):`);
        if (password === null) return;
        if (password.length < 12) throw new Error('Password must be at least 12 characters.');
        await api(`/api/v1/auth/users/${userAction.dataset.userId}/reset-password`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({password})});
      }
      await Promise.all([loadUsers(), loadDirectory()]);
    } catch (error) { $('userStatus').className = 'status-message error'; $('userStatus').textContent = error.message; }
  }
});

boot();

