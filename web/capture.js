/* Capture client.
 *
 * The whole reason this exists instead of a file picker: a photo taken with the
 * native camera app lands in the camera roll and cannot be deleted
 * programmatically. Capturing from a MediaStream into a canvas means the image
 * only ever exists as a Blob in this tab.
 *
 * The one rule that shapes everything below: a captured Blob is held in memory
 * until the server reports `done`. The server deletes the page image the moment
 * extraction succeeds, so until then this tab holds the only other copy — and
 * dropping it early turns any failure into re-shooting the page.
 */

const MAX_EDGE = 2000;   // matches app/extraction/images.py
const QUALITY = 0.8;
const POLL_MS = 3000;

const el = (id) => document.getElementById(id);
const state = {
  token: sessionStorage.getItem('token') || '',
  sessionId: null,
  pages: [],      // {blob, url} for the recipe currently being shot
  tracked: [],    // {recipeId, title, status, pages, startedAt, elapsed}
  stream: null,
};

/* ---------- api ---------- */

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { Authorization: `Bearer ${state.token}`, ...(options.headers || {}) },
  });
  if (response.status === 401) {
    signOut('That token was rejected.');
    throw new Error('unauthorized');
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

/* ---------- auth ---------- */

function signOut(message) {
  // sessionStorage, not localStorage: a token that outlives the tab is a
  // credential sitting on a phone that might be lost or handed to someone.
  sessionStorage.removeItem('token');
  state.token = '';
  stopCamera();
  el('app').hidden = true;
  el('gate').hidden = false;
  if (message) {
    el('gate-error').textContent = message;
    el('gate-error').hidden = false;
  }
}

el('gate-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  state.token = el('token-input').value.trim();
  el('gate-error').hidden = true;
  try {
    await api('/api/recipes?limit=1');
  } catch {
    return;   // signOut already surfaced the message
  }
  sessionStorage.setItem('token', state.token);
  el('token-input').value = '';
  start();
});

el('sign-out').addEventListener('click', () => signOut(''));

/* ---------- camera ---------- */

async function startCamera() {
  // Requires a secure context. Over plain HTTP on a LAN address this rejects,
  // and the file fallback is the only route.
  if (!navigator.mediaDevices?.getUserMedia) {
    return showCameraFallback('This browser has no camera API available.');
  }

  // A pending permission prompt never settles this promise. Without a nudge the
  // user is left looking at a black rectangle with nothing to act on.
  const nudge = setTimeout(
    () => offerFallback('Waiting for camera permission — or use the button below.'),
    8000,
  );

  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 3000 }, height: { ideal: 3000 } },
      audio: false,
    });
    el('preview').srcObject = state.stream;
    el('camera-error').hidden = true;
  } catch (error) {
    showCameraFallback(`Camera unavailable (${error.name}). Use the button below.`);
  } finally {
    clearTimeout(nudge);
  }
}

/* Additive: the camera may still be coming. */
function offerFallback(message) {
  el('camera-error').textContent = message;
  el('camera-error').hidden = false;
  el('file-fallback').hidden = false;
}

/* Terminal: there will be no stream, so the shutter is dead weight. */
function showCameraFallback(message) {
  offerFallback(message);
  el('shutter').hidden = true;
}

function stopCamera() {
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
}

/* ---------- capture ---------- */

/* Downscale and encode. Mirrors prepare_image() server-side so the quality
 * measured during the Phase 1 spike is the quality this actually sends.
 * No thresholding or deskewing: those help classic OCR and hurt vision models. */
function encode(source, width, height) {
  const scale = Math.min(1, MAX_EDGE / Math.max(width, height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  canvas.getContext('2d').drawImage(source, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', QUALITY));
}

function addPage(blob) {
  state.pages.push({ blob, url: URL.createObjectURL(blob) });
  renderFilmstrip();
}

el('shutter').addEventListener('click', async () => {
  const video = el('preview');
  if (!video.videoWidth) return;
  addPage(await encode(video, video.videoWidth, video.videoHeight));
});

el('file-input').addEventListener('change', async (event) => {
  for (const file of event.target.files) {
    const bitmap = await createImageBitmap(file);
    addPage(await encode(bitmap, bitmap.width, bitmap.height));
    bitmap.close();
  }
  event.target.value = '';
});

function clearPages() {
  state.pages.forEach((page) => URL.revokeObjectURL(page.url));
  state.pages = [];
  renderFilmstrip();
}

el('discard').addEventListener('click', clearPages);

function renderFilmstrip() {
  const strip = el('filmstrip');
  strip.hidden = state.pages.length === 0;
  strip.innerHTML = '';
  state.pages.forEach((page, index) => {
    const figure = document.createElement('figure');
    figure.innerHTML = `<img src="${page.url}" alt="Page ${index + 1}"><figcaption>${index + 1}</figcaption>`;
    const remove = document.createElement('button');
    remove.textContent = '×';
    remove.title = 'Remove this page';
    remove.addEventListener('click', () => {
      URL.revokeObjectURL(page.url);
      state.pages.splice(index, 1);
      renderFilmstrip();
    });
    figure.appendChild(remove);
    strip.appendChild(figure);
  });

  el('page-count').textContent = state.pages.length ? `(${state.pages.length})` : '';
  el('submit').disabled = state.pages.length === 0;
  el('discard').disabled = state.pages.length === 0;
}

/* ---------- upload ---------- */

async function ensureSession() {
  if (state.sessionId) return state.sessionId;
  const form = new FormData();
  form.append('book_title', el('book-title').value.trim());
  state.sessionId = (await api('/api/sessions', { method: 'POST', body: form })).session_id;
  return state.sessionId;
}

function buildForm(pages, hint, sessionId) {
  const form = new FormData();
  pages.forEach((page, index) => form.append('images', page.blob, `${index}.jpg`));
  form.append('session_id', sessionId);
  form.append('hint', hint);
  return form;
}

el('submit').addEventListener('click', async () => {
  if (!state.pages.length) return;

  // Handed to the tracker, which owns them until the server confirms `done`.
  const pages = state.pages;
  const hint = el('hint').value.trim();
  state.pages = [];
  renderFilmstrip();
  el('hint').value = '';

  const entry = {
    recipeId: null,
    title: hint || `${pages.length} page${pages.length > 1 ? 's' : ''}`,
    status: 'uploading',
    pages,
    startedAt: Date.now(),
    error: '',
  };
  state.tracked.unshift(entry);
  renderQueue();

  await upload(entry, hint);
});

async function upload(entry, hint) {
  try {
    const sessionId = await ensureSession();
    const result = await api('/api/recipes', {
      method: 'POST',
      body: buildForm(entry.pages, hint, sessionId),
    });
    entry.recipeId = result.recipe_id;
    entry.status = 'pending';
    entry.hint = hint;
  } catch (error) {
    // The blobs are still held, so this is recoverable without re-shooting.
    entry.status = 'upload-failed';
    entry.error = error.message;
  }
  renderQueue();
}

/* ---------- polling ---------- */

async function poll() {
  const pending = state.tracked.filter((e) => e.recipeId && !['done', 'failed'].includes(e.status));
  for (const entry of pending) {
    try {
      const recipe = await api(`/api/recipes/${entry.recipeId}`);
      entry.status = recipe.status;
      entry.error = recipe.error || '';
      if (recipe.title) entry.title = recipe.title;
      entry.confidence = recipe.confidence;
      entry.issues = recipe.review?.issues?.length ?? 0;
      if (recipe.status === 'done') releasePages(entry);
    } catch (error) {
      entry.error = error.message;
    }
  }
  if (state.tracked.length) renderQueue();
}

/* The server has the transcription committed and has deleted its own copy of
 * the image, so this is the moment the tab's copy stops being the last one. */
function releasePages(entry) {
  entry.pages.forEach((page) => URL.revokeObjectURL(page.url));
  entry.pages = [];
}

async function retry(entry) {
  entry.error = '';
  if (entry.status === 'upload-failed') {
    entry.status = 'uploading';
    renderQueue();
    return upload(entry, entry.hint || '');
  }
  if (!entry.pages.length) {
    // Server-side images are gone too; nothing left to resend.
    entry.error = 'Page images are gone. This page must be re-shot.';
    renderQueue();
    return;
  }
  try {
    await api(`/api/recipes/${entry.recipeId}/retry`, { method: 'POST' });
    entry.status = 'pending';
  } catch (error) {
    entry.error = error.message;
  }
  renderQueue();
}

/* ---------- queue ---------- */

const LABEL = {
  uploading: 'Uploading',
  'upload-failed': 'Upload failed',
  pending: 'Queued',
  running: 'Extracting',
  done: 'Done',
  failed: 'Failed',
};

function renderQueue() {
  const queue = el('queue');
  queue.innerHTML = '';
  for (const entry of state.tracked) {
    const item = document.createElement('article');
    item.className = `queue-item ${entry.status}`;

    const seconds = Math.round((Date.now() - entry.startedAt) / 1000);
    const busy = ['uploading', 'pending', 'running'].includes(entry.status);
    const detail = entry.status === 'done'
      ? `${entry.confidence || '?'} confidence${entry.issues ? `, ${entry.issues} to review` : ''}`
      : busy ? `${seconds}s` : entry.error;

    item.innerHTML = `
      <div class="queue-main">
        <strong>${escapeHtml(entry.title)}</strong>
        <span class="status">${LABEL[entry.status] || entry.status}</span>
      </div>
      <div class="queue-detail">${escapeHtml(detail || '')}</div>`;

    if (['failed', 'upload-failed'].includes(entry.status)) {
      const button = document.createElement('button');
      button.textContent = entry.pages.length ? 'Retry' : 'Dismiss';
      button.addEventListener('click', () => {
        if (entry.pages.length) return retry(entry);
        state.tracked = state.tracked.filter((e) => e !== entry);
        renderQueue();
      });
      item.appendChild(button);
    }
    queue.appendChild(item);
  }
}

function escapeHtml(text) {
  const node = document.createElement('span');
  node.textContent = text;
  return node.innerHTML;
}

/* ---------- lifecycle ---------- */

// Anything still held here has not been confirmed extracted, and the server may
// have nothing usable either.
window.addEventListener('beforeunload', (event) => {
  const held = state.pages.length + state.tracked.reduce((n, e) => n + e.pages.length, 0);
  if (held > 0) event.preventDefault();
});

function start() {
  el('gate').hidden = true;
  el('app').hidden = false;
  startCamera();
  renderFilmstrip();
  setInterval(poll, POLL_MS);
}

if (state.token) start();
