const element = (id) => document.getElementById(id);
const imageRoles = ['front', 'right', 'left', 'back'];
const statusNames = {
  queued: 'Queued', running: 'Running', succeeded: 'Complete', failed: 'Failed', interrupted: 'Interrupted',
};
const dates = new Intl.DateTimeFormat('en', { dateStyle: 'medium', timeStyle: 'short' });
const numbers = new Intl.NumberFormat('en');
const historyList = element('history-list');
const modelVariant = element('model-variant');
const rotateControl = element('auto-rotate');
const modeButtons = [...element('display-modes').querySelectorAll('button')];
const deleteDialog = element('delete-dialog');
const runLog = element('run-log');
const logOutput = element('log-output');

let runs = [];
let selectedId = null;
let selectedRun = null;
let historyRequest = 0;
let selectionRequest = 0;
let modelRequest = 0;
let pollTimer;
let submitting = false;
let displayMode = 'textured';
let requestedModel = null;
let loadedRun = null;
let viewer;
let viewerImport;
let deleteTarget = null;
let deleting = false;
let dragDepth = 0;
let dropSlot = null;

const isActive = (run) => run?.status === 'running' || run?.status === 'queued';
const isWebJobActive = (run) => run?.origin === 'web' && isActive(run);
const capitalize = (value) => value.charAt(0).toUpperCase() + value.slice(1);
const localUrl = (value) => typeof value === 'string' && /^\/(?![\\/])/.test(value) ? value : null;

function setMessage(target, message) {
  if (target.textContent !== message) target.textContent = message;
  target.hidden = !message;
}

function showStatus(target, status) {
  target.dataset.status = status ?? '';
  target.textContent = statusNames[status] ?? '';
}

function formatDate(value) {
  const date = new Date(value);
  return value && !Number.isNaN(date.getTime()) ? dates.format(date) : '';
}

async function request(url, options) {
  const response = await fetch(url, { cache: 'no-store', ...options });
  if (response.status === 204) return null;
  const data = await response.json().catch(() => null);
  if (!response.ok || !data) {
    throw new Error(data?.error || `The local server returned HTTP ${response.status}.`);
  }
  return data;
}

function assignPhotos(files, role) {
  const message = element('generation-error');
  const feedback = element('photo-feedback');
  setMessage(message, '');
  setMessage(feedback, '');
  if (!files.length) return;
  if (files.length !== 1 && files.length !== imageRoles.length) {
    setMessage(message, 'Choose one front photo or all four photos. Use the individual slots to add other angles one at a time.');
    return;
  }
  if (files.some((file) => !/\.(?:png|jpe?g|jpe|webp|gif|bmp)$/i.test(file.name))) {
    setMessage(message, 'Choose PNG, JPEG, WebP, GIF or BMP images.');
    return;
  }
  if (files.some((file) => file.size > 24 * 1024 * 1024)) {
    setMessage(message, 'Each image must be 24 MiB or smaller.');
    return;
  }

  const roles = files.length === 1 ? [role ?? 'front'] : imageRoles;
  for (const [index, current] of roles.entries()) {
    const input = element(current);
    const transfer = new DataTransfer();
    transfer.items.add(files[index]);
    input.files = transfer.files;
  }
  if (roles.some((current) => current !== 'front')) element('extra-views').open = true;
  setMessage(feedback, files.length === 1
    ? `Added ${files[0].name} to ${capitalize(roles[0])}.`
    : 'Added four photos. Check each angle before generating.');
}

function draggedFile(event) {
  return Array.from(event.dataTransfer?.types ?? []).includes('Files');
}

function slotAt(target) {
  return target instanceof Element ? target.closest('[data-photo-role]') : null;
}

function markDropSlot(slot) {
  dropSlot?.classList.remove('is-drop-target');
  dropSlot = slot;
  dropSlot?.classList.add('is-drop-target');
  element('drop-indicator').textContent = slot
    ? `Drop photo into ${capitalize(slot.dataset.photoRole)}`
    : 'Drop photos to add them';
}

function clearDrag() {
  dragDepth = 0;
  document.body.classList.remove('dragging-files');
  markDropSlot(null);
}

function updateGenerateButton() {
  const busy = isWebJobActive(selectedRun) || runs.some(isWebJobActive);
  const button = element('generate-button');
  button.disabled = submitting || busy;
  button.textContent = submitting ? 'Submitting…' : busy ? 'Generation in progress' : 'Generate model';
}

function updateDeleteButton() {
  const button = element('delete-run');
  button.hidden = !selectedRun;
  button.disabled = isActive(selectedRun) || deleting;
  button.title = isActive(selectedRun) ? 'Wait for the generation to finish' : 'Delete this generation from history';
}

function renderHistory() {
  const activeButton = document.activeElement?.closest('[data-run-id]');
  const fragment = document.createDocumentFragment();
  for (const run of runs) {
    const entry = document.createElement('li');
    const button = document.createElement('button');
    button.className = 'history-item';
    button.type = 'button';
    button.dataset.runId = run.id;
    button.setAttribute('aria-current', String(run.id === selectedId));

    const thumbnail = localUrl(run.thumbnail_url);
    if (thumbnail) {
      const image = document.createElement('img');
      image.className = 'history-thumb';
      image.src = thumbnail;
      image.alt = '';
      image.loading = 'lazy';
      button.append(image);
    }

    const description = document.createElement('span');
    description.className = 'history-desc';
    const name = document.createElement('span');
    name.className = 'history-name';
    name.textContent = run.name || 'Generation';
    name.title = name.textContent;
    const date = document.createElement('span');
    date.className = 'history-date';
    date.textContent = formatDate(run.created_at);
    description.append(name, date);

    const status = document.createElement('span');
    status.className = 'status';
    showStatus(status, run.status);
    button.append(description, status);
    entry.append(button);
    fragment.append(entry);
  }
  historyList.replaceChildren(fragment);
  if (activeButton && !activeButton.isConnected) {
    [...historyList.querySelectorAll('button')]
      .find((button) => button.dataset.runId === activeButton.dataset.runId)?.focus({ preventScroll: true });
  }
}

function renderSources(run) {
  const sources = element('source-list');
  const fragment = document.createDocumentFragment();
  for (const role of imageRoles) {
    const source = run.images?.find((image) => image.role === role);
    const url = localUrl(source?.url);
    if (!url) continue;
    const entry = document.createElement('li');
    const link = document.createElement('a');
    link.className = 'source-image';
    link.href = url;
    link.target = '_blank';
    link.rel = 'noopener';
    const img = document.createElement('img');
    img.src = url;
    img.alt = '';
    img.loading = 'lazy';
    const roleLabel = document.createElement('span');
    roleLabel.className = 'source-role';
    roleLabel.textContent = capitalize(role);
    const name = document.createElement('span');
    name.className = 'source-name';
    name.textContent = source.name;
    name.title = source.name;
    link.append(img, roleLabel, name);
    entry.append(link);
    fragment.append(entry);
  }
  setMessage(element('sources-message'), fragment.childElementCount ? '' : 'No source images saved for this generation.');
  sources.replaceChildren(fragment);
}

function updateLog(run, differentRun) {
  if (differentRun) runLog.open = false;
  const active = isActive(run);
  const text = run.log || (active ? 'Waiting for generation output…' : '');
  const scrollTop = logOutput.scrollTop;
  const following = logOutput.scrollHeight - scrollTop - logOutput.clientHeight <= 10;
  runLog.hidden = !text;
  if (logOutput.textContent !== text) {
    const previous = logOutput.firstChild;
    if (!differentRun && previous instanceof Text && text.startsWith(previous.data)) {
      previous.appendData(text.slice(previous.length));
    } else {
      logOutput.textContent = text;
    }
    if (runLog.open) logOutput.scrollTop = following ? logOutput.scrollHeight : scrollTop;
  }
}

function renderRun(run) {
  const previous = selectedRun;
  selectedRun = run;
  setMessage(element('selection-title'), run.name || '');
  element('run-detail').hidden = false;
  showStatus(element('run-status'), run.status);
  const active = isActive(run);
  const progress = Number.isFinite(run.progress) ? Math.max(0, Math.min(run.progress, 1)) : null;
  const stage = active ? [run.stage, progress === null ? '' : `${Math.round(progress * 100)}%`].filter(Boolean).join(' · ') : '';
  setMessage(element('run-stage'), stage);
  const meter = element('run-progress');
  meter.hidden = !active;
  if (progress === null) meter.removeAttribute('value');
  else meter.value = progress;
  setMessage(element('run-error'), run.error || '');

  const stats = [
    ['Quality', run.tier ? capitalize(run.tier) : null],
    ['Seed', Number.isInteger(run.seed) && run.seed >= 0 ? numbers.format(run.seed) : null],
    ['Faces', Number.isFinite(run.faces) ? numbers.format(run.faces) : null],
    ['Time', Number.isFinite(run.duration_s) ? `${Math.round(run.duration_s)} s` : null],
    ['Peak VRAM', Number.isFinite(run.peak_vram_gb) ? `${run.peak_vram_gb.toFixed(2)} GB` : null],
  ];
  const metadata = element('run-metadata');
  metadata.replaceChildren();
  for (const [label, value] of stats) {
    if (value === null) continue;
    const item = document.createElement('div');
    const term = document.createElement('dt');
    term.textContent = label;
    const definition = document.createElement('dd');
    definition.textContent = value;
    item.append(term, definition);
    metadata.append(item);
  }
  metadata.hidden = metadata.childElementCount === 0;

  const differentRun = run.id !== previous?.id;
  if (differentRun) modelVariant.value = 'final';
  updateLog(run, differentRun);

  if (differentRun || JSON.stringify(run.images) !== JSON.stringify(previous?.images)) renderSources(run);
  updateGenerateButton();
  updateDeleteButton();
  void updateModel(run);
}

function setViewerControls(enabled) {
  for (const button of modeButtons) button.disabled = !enabled;
  rotateControl.disabled = !enabled;
  element('viewer-help').hidden = !enabled;
}

async function updateModel(run) {
  const finalUrl = localUrl(run.model_url);
  const aiUrl = localUrl(run.ai_model_url);
  const variant = modelVariant.value === 'ai' ? 'ai' : 'final';
  element('variant-control').hidden = !finalUrl || !aiUrl;
  const url = (variant === 'ai' ? aiUrl : finalUrl) ?? finalUrl ?? aiUrl;
  const viewerMessage = element('viewer-message');
  const download = element('download-model');
  download.hidden = !url;
  if (url) {
    const filename = Number.isInteger(run.seed) && run.seed >= 0 ? `model_${run.seed}` : 'model';
    download.href = url;
    download.download = `${filename}${url === aiUrl ? '.ai' : ''}.glb`;
  } else {
    download.removeAttribute('href');
  }

  if (!url) {
    if (requestedModel !== null) {
      modelRequest++;
      requestedModel = null;
      loadedRun = null;
      viewer?.clear();
    }
    setViewerControls(false);
    setMessage(viewerMessage, isActive(run) ? 'The model will appear here when generation finishes.' : 'No model is available for this generation.');
    return;
  }
  if (url === requestedModel) return;

  requestedModel = url;
  const requestId = ++modelRequest;
  setViewerControls(false);
  setMessage(viewerMessage, 'Loading model…');
  if (loadedRun !== run.id) viewer?.clear();
  try {
    if (!viewer) {
      viewerImport ??= import('/static/viewer.js');
      const { MeshViewer } = await viewerImport;
      if (requestId !== modelRequest) return;
      viewer = new MeshViewer(element('viewer-surface'));
      viewer.setMode(displayMode);
      viewer.setAutoRotate(rotateControl.checked);
    }
  } catch {
    if (requestId === modelRequest) setMessage(viewerMessage, '3D preview is unavailable. You can still download the GLB.');
    return;
  }
  try {
    const loaded = await viewer.load(url, loadedRun !== run.id);
    if (!loaded || requestId !== modelRequest) return;
    loadedRun = run.id;
    setViewerControls(true);
    setMessage(viewerMessage, '');
  } catch {
    if (requestId !== modelRequest) return;
    viewer.clear();
    loadedRun = null;
    setMessage(viewerMessage, 'This GLB could not be displayed. You can still download it.');
  }
}

function schedulePoll() {
  clearTimeout(pollTimer);
  if (isActive(selectedRun) || runs.some(isActive)) {
    pollTimer = setTimeout(refreshHistory, 1800);
  }
}

function clearSelection() {
  selectionRequest++;
  selectedId = selectedRun = null;
  element('run-detail').hidden = true;
  setMessage(element('selection-title'), '');
  setMessage(element('sources-message'), 'Select a generation to see its source images.');
  element('source-list').replaceChildren();
  void updateModel({});
  setMessage(element('viewer-message'), 'Generate a model to preview it here.');
  updateDeleteButton();
}

async function selectRun(id) {
  const changed = selectedId !== id;
  selectedId = id;
  const requestId = ++selectionRequest;
  if (changed) {
    for (const button of historyList.querySelectorAll('button')) {
      button.setAttribute('aria-current', String(button.dataset.runId === id));
    }
    const summary = runs.find((run) => run.id === id);
    if (summary) renderRun(summary);
  }
  try {
    const run = await request(`/api/runs/${encodeURIComponent(id)}`);
    if (requestId === selectionRequest) renderRun(run);
  } catch (error) {
    if (requestId === selectionRequest) setMessage(element('history-message'), error.message);
  }
}

async function refreshHistory() {
  clearTimeout(pollTimer);
  const requestId = ++historyRequest;
  const refresh = element('refresh-history');
  refresh.disabled = true;
  try {
    const data = await request('/api/runs');
    if (requestId !== historyRequest) return;
    const signature = (list) => JSON.stringify(list.map((run) => [run.id, run.name, run.status, run.thumbnail_url, run.created_at]));
    const changed = signature(runs) !== signature(data.runs);
    runs = data.runs;
    if (changed) renderHistory();
    setMessage(element('history-message'), runs.length ? '' : 'No generations yet.');
    const current = runs.find((run) => run.id === selectedId);
    const next = current ?? runs[0];
    if (next) {
      await selectRun(next.id);
    } else clearSelection();
  } catch (error) {
    if (requestId !== historyRequest) return;
    setMessage(element('history-message'), error.message);
    if (!selectedId) setMessage(element('viewer-message'), 'Generation history could not be loaded.');
  } finally {
    if (requestId === historyRequest) {
      refresh.disabled = false;
      updateGenerateButton();
      schedulePoll();
    }
  }
}

const photoPicker = element('photo-picker');
runLog.addEventListener('toggle', () => {
  if (!runLog.open || runLog.hidden) return;
  logOutput.scrollTop = logOutput.scrollHeight;
  runLog.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'instant' });
});
element('choose-photos').addEventListener('click', () => photoPicker.click());
photoPicker.addEventListener('change', () => {
  assignPhotos([...photoPicker.files]);
  photoPicker.value = '';
});
for (const role of imageRoles) {
  element(role).addEventListener('change', () => {
    setMessage(element('generation-error'), '');
    setMessage(element('photo-feedback'), '');
  });
}

window.addEventListener('dragenter', (event) => {
  if (!draggedFile(event)) return;
  event.preventDefault();
  if (deleteDialog.open) return;
  dragDepth++;
  document.body.classList.add('dragging-files');
});
window.addEventListener('dragover', (event) => {
  if (!draggedFile(event)) return;
  event.preventDefault();
  event.dataTransfer.dropEffect = deleteDialog.open ? 'none' : 'copy';
  if (deleteDialog.open) return;
  const slot = event.dataTransfer.items.length === 1 ? slotAt(event.target) : null;
  if (slot !== dropSlot) markDropSlot(slot);
});
window.addEventListener('dragleave', () => {
  if (dragDepth > 0 && --dragDepth === 0) clearDrag();
});
window.addEventListener('drop', (event) => {
  if (!draggedFile(event)) return;
  event.preventDefault();
  clearDrag();
  if (deleteDialog.open) return;
  assignPhotos([...event.dataTransfer.files], slotAt(event.target)?.dataset.photoRole);
  element('generation-heading').scrollIntoView({ block: 'nearest' });
});
window.addEventListener('blur', clearDrag);

element('generation-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const message = element('generation-error');
  setMessage(message, '');
  const files = imageRoles.map((role) => element(role).files[0]);
  const extra = files.slice(1).filter(Boolean);
  if (extra.length > 0 && extra.length < 3) {
    element('extra-views').open = true;
    setMessage(message, 'Add the right, left and back images for four-view input.');
    return;
  }
  if (files.some((file) => file?.size > 24 * 1024 * 1024)) {
    setMessage(message, 'Each image must be 24 MiB or smaller.');
    return;
  }
  const upload = new FormData();
  for (const [index, file] of files.entries()) {
    if (file) upload.append(imageRoles[index], file);
  }
  upload.append('tier', element('tier').value);
  upload.append('seed', element('seed').value || '-1');
  upload.append('projection', String(element('projection').checked));

  submitting = true;
  updateGenerateButton();
  try {
    const run = await request('/api/runs', { method: 'POST', body: upload });
    historyRequest++;
    selectionRequest++;
    element('refresh-history').disabled = false;
    runs = [run, ...runs.filter(({ id }) => id !== run.id)];
    selectedId = run.id;
    setMessage(element('history-message'), '');
    renderHistory();
    renderRun(run);
    for (const role of imageRoles) element(role).value = '';
    element('extra-views').open = false;
    setMessage(element('photo-feedback'), '');
  } catch (error) {
    setMessage(message, error.message);
  } finally {
    submitting = false;
    updateGenerateButton();
    schedulePoll();
  }
});

element('delete-run').addEventListener('click', () => {
  if (!selectedRun || isActive(selectedRun) || deleting) return;
  deleteTarget = selectedRun.id;
  const caseDescription = selectedRun.legacy
    ? 'The model and any matching AI version or preview thumbnail will be permanently deleted from the output folder.'
    : 'This permanently removes the run, its models, preview and log. Matching output files linked only to this run are also removed. Deletion covers only the source-photo copies stored in history.';
  setMessage(element('delete-target'), selectedRun.name || 'Generation');
  setMessage(element('delete-description'), `${caseDescription} Original source images outside the history are kept.`);
  setMessage(element('delete-error'), '');
  clearDrag();
  deleteDialog.showModal();
});

element('cancel-delete').addEventListener('click', () => deleteDialog.close());
deleteDialog.addEventListener('cancel', (event) => {
  if (deleting) event.preventDefault();
});
deleteDialog.addEventListener('close', () => {
  deleteTarget = null;
});
element('confirm-delete').addEventListener('click', async () => {
  if (!deleteTarget || deleting) return;
  const id = deleteTarget;
  const confirm = element('confirm-delete');
  const cancel = element('cancel-delete');
  const refresh = element('refresh-history');
  deleting = confirm.disabled = cancel.disabled = true;
  confirm.textContent = 'Deleting…';
  setMessage(element('delete-error'), '');
  updateDeleteButton();
  try {
    await request(`/api/runs/${encodeURIComponent(id)}`, { method: 'DELETE' });
    historyRequest++;
    runs = runs.filter((run) => run.id !== id);
    if (selectedId === id) {
      const next = runs[0];
      if (next) {
        selectionRequest++;
        selectedId = next.id;
        renderRun(next);
      } else clearSelection();
    }
    renderHistory();
    deleteDialog.close();
    await refreshHistory();
    (historyList.querySelector('[aria-current="true"]') ?? refresh).focus({ preventScroll: true });
  } catch (error) {
    setMessage(element('delete-error'), error.message);
  } finally {
    deleting = confirm.disabled = cancel.disabled = false;
    confirm.textContent = 'Delete generation';
    updateDeleteButton();
  }
});

element('refresh-history').addEventListener('click', refreshHistory);
historyList.addEventListener('click', (event) => {
  const button = event.target.closest('[data-run-id]');
  if (!button) return;
  void selectRun(button.dataset.runId).then(schedulePoll);
});

for (const button of modeButtons) {
  button.addEventListener('click', () => {
    displayMode = button.dataset.mode;
    for (const mode of modeButtons) mode.setAttribute('aria-pressed', String(mode === button));
    viewer.setMode(displayMode);
  });
}

rotateControl.addEventListener('change', () => viewer.setAutoRotate(rotateControl.checked));
modelVariant.addEventListener('change', () => {
  if (selectedRun) void updateModel(selectedRun);
});

void refreshHistory();
