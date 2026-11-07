/* Real-time next-word autocomplete.
 *
 * Flow: input -> debounce -> fetch('/api/predict') -> render chips + inline ghost text.
 * Stale responses are ignored twice over: the previous request is aborted AND every response is
 * checked against a monotonically increasing request id and the text it was computed for.
 * Accepting a suggestion edits the textarea programmatically (which does not fire 'input'),
 * then requests ONE new prediction, so there is no feedback loop and no duplicated text.
 */
(function () {
  'use strict';

  const DEBOUNCE_MS = 180;
  const cfg = window.NW_CONFIG || { ready: false, models: [] };

  const editor = document.getElementById('editor');
  const backdrop = document.getElementById('backdrop');
  const chipsEl = document.getElementById('chips');
  const statusEl = document.getElementById('status');
  const statusRow = statusEl.parentElement;
  const spinner = document.getElementById('spinner');
  const topkSel = document.getElementById('topk');
  const modelButtons = Array.from(document.querySelectorAll('#model-switch .seg'));

  const ACRONYMS = new Set(['rnn', 'lstm', 'gru', 'cnn', 'gpu', 'gpus', 'cpu', 'tpu', 'nlp', 'gan', 'gans', 'bert', 'gpt',
    'api', 'mlp', 'ai', 'sgd', 'llm', 'llms', 'vae', 'ocr', 'pca', 'dnn', 'ann', 'rl', 'rlhf', 'rag', 'tf', 'nn']);

  const state = {
    model: null,
    requestId: 0,
    controller: null,
    timer: null,
    composing: false,
    predictions: [],      // [{word, probability}] for `forText`
    mode: 'next_word',
    prefix: '',
    forText: null,        // text-before-caret the current predictions belong to
    forModel: null,
    lastStatus: 'ok',     // status of the response for forText ('ok', 'unknown_words', ...)
    dismissedFor: null,   // Esc pressed: hide ghost until the text changes
  };

  /* ---------------------------------------------------------------- helpers */
  const escapeHtml = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  function caretInfo() {
    const pos = editor.selectionStart;
    const collapsed = editor.selectionStart === editor.selectionEnd;
    return { before: editor.value.slice(0, pos), after: editor.value.slice(pos), collapsed };
  }

  function atEnd() {
    const c = caretInfo();
    return c.collapsed && c.after === '';
  }

  /* Capitalise at the start of a sentence; upper-case well-known acronyms. Used for whole next words only. */
  function formatWord(word, before) {
    let w = ACRONYMS.has(word) ? word.toUpperCase() : word;
    if (before.trim() === '' || /[.!?]\s*$/.test(before)) {
      w = w.charAt(0).toUpperCase() + w.slice(1);
    }
    return w;
  }

  /* What exactly would be inserted for a prediction, given the text before the caret. */
  function insertionFor(pred, before, mode, prefix) {
    if (mode === 'complete_word') {
      return pred.word.slice(prefix.length);                 // finish the fragment the user already typed
    }
    const needsSpace = before.length > 0 && !/\s$/.test(before);
    return (needsSpace ? ' ' : '') + formatWord(pred.word, before);
  }

  /* ---------------------------------------------------------------- rendering */
  function setStatus(text, isError) {
    statusEl.textContent = text;
    statusRow.classList.toggle('error', !!isError);
  }

  function setLoading(on) {
    spinner.hidden = !on;
    chipsEl.classList.toggle('stale', on && state.predictions.length > 0);
  }

  function renderGhost() {
    const text = editor.value;
    let ghost = '';
    const top = state.predictions[0];
    if (top && state.forText !== null && state.forModel === state.model && atEnd()
        && caretInfo().before === state.forText && state.dismissedFor !== state.forText) {
      ghost = insertionFor(top, state.forText, state.mode, state.prefix);
    }
    // trailing zero-width space keeps the last (possibly empty) line height identical to the textarea
    backdrop.innerHTML = '<span class="typed">' + escapeHtml(text) + '</span>' +
      (ghost ? '<span class="ghost">' + escapeHtml(ghost) + '</span>' : '') + '\u200b';
    return ghost !== '';
  }

  function renderChips() {
    chipsEl.innerHTML = '';
    chipsEl.classList.remove('stale');
    if (!state.predictions.length) {
      return;
    }
    const before = state.forText || '';
    state.predictions.forEach((pred, i) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip' + (i === 0 ? ' top' : '');
      b.setAttribute('role', 'listitem');
      const label = state.mode === 'complete_word' ? pred.word : formatWord(pred.word, before);
      const w = document.createElement('span');
      w.className = 'w';
      w.textContent = label;
      const p = document.createElement('span');
      p.className = 'p';
      const pct = pred.probability * 100;
      p.textContent = pct >= 0.1 ? pct.toFixed(1) + '%' : '<0.1%';
      b.append(w, p);
      if (i === 0) {
        const k = document.createElement('kbd');
        k.textContent = 'Tab';
        b.append(k);
      }
      b.addEventListener('mousedown', (e) => e.preventDefault());   // keep focus + caret in the editor
      b.addEventListener('click', () => acceptPrediction(pred));
      chipsEl.append(b);
    });
  }

  function clearSuggestions() {
    state.predictions = [];
    state.forText = null;
    renderChips();
    renderGhost();
  }

  /* ---------------------------------------------------------------- prediction requests */
  function schedulePrediction(delay) {
    clearTimeout(state.timer);
    state.timer = setTimeout(requestPrediction, delay === undefined ? DEBOUNCE_MS : delay);
  }

  function cancelInFlight() {
    clearTimeout(state.timer);
    state.requestId += 1;                  // any response already on its way is now stale
    if (state.controller) {
      state.controller.abort();
      state.controller = null;
    }
  }

  async function requestPrediction() {
    if (!cfg.ready || state.composing) return;
    const { before, collapsed } = caretInfo();
    if (!collapsed) return;                // text is selected: wait for the user

    if (before.trim() === '') {
      cancelInFlight();
      setLoading(false);
      clearSuggestions();
      setStatus('Predictions appear automatically as you type.');
      return;
    }

    const topK = Number(topkSel.value) || 5;
    // Same text + model as the predictions we already show: nothing to fetch, just re-render.
    if (state.forText === before && state.forModel === state.model
        && (state.predictions.length >= topK || state.lastStatus !== 'ok')) {
      renderGhost();
      return;
    }

    cancelInFlight();
    const id = state.requestId;
    const controller = new AbortController();
    state.controller = controller;
    const model = state.model;
    setLoading(true);
    setStatus('Predicting with ' + model + '…');

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: before, model: model, top_k: topK }),
        signal: controller.signal,
      });
      let data = null;
      try { data = await res.json(); } catch (_) { /* non-JSON error page */ }

      if (id !== state.requestId) return;                                   // superseded
      if (caretInfo().before !== before || model !== state.model) return;   // text/model changed meanwhile

      setLoading(false);
      if (!res.ok || !data || data.status === 'error') {
        clearSuggestions();
        setStatus((data && data.error) || 'The server could not make a prediction.', true);
        return;
      }
      if (data.model !== model) {            // defensive: the server must have used the model we asked for
        clearSuggestions();
        setStatus('Unexpected response from server.', true);
        return;
      }

      state.predictions = data.status === 'ok' ? data.predictions : [];
      state.mode = data.mode;
      state.prefix = data.prefix || '';
      state.forText = before;
      state.forModel = model;
      state.lastStatus = data.status;
      state.dismissedFor = null;
      renderChips();
      renderGhost();

      if (data.status === 'ok') {
        const extra = data.unknown_in_context ? ' · ' + data.unknown_in_context + ' unfamiliar word(s) ignored' : '';
        setStatus(model + ' · ' + Number(data.duration_ms).toFixed(0) + ' ms' + extra);
      } else {
        setStatus(data.message || 'No suggestion for this text.');
      }
    } catch (err) {
      if (err && err.name === 'AbortError') return;                         // replaced by a newer request
      if (id !== state.requestId) return;
      setLoading(false);
      clearSuggestions();
      setStatus('Could not reach the server. Check that it is running.', true);
    } finally {
      if (state.controller === controller && id === state.requestId) state.controller = null;
    }
  }

  /* ---------------------------------------------------------------- accepting suggestions */
  function acceptPrediction(pred) {
    if (!pred || state.forText === null) return;
    const { before, after, collapsed } = caretInfo();
    if (!collapsed || before !== state.forText) return;     // predictions are for different text: refuse to guess

    let insertion = insertionFor(pred, before, state.mode, state.prefix) + ' ';
    if (/^\s/.test(after)) insertion = insertion.trimEnd();  // never create a double space
    editor.value = before + insertion + after;
    const caret = (before + insertion).length;
    editor.setSelectionRange(caret, caret);
    editor.focus();

    // Programmatic edits do not fire 'input', so request exactly one fresh prediction ourselves.
    cancelInFlight();
    state.predictions = [];
    state.forText = null;
    renderChips();
    renderGhost();
    requestPrediction();
  }

  /* ---------------------------------------------------------------- events */
  editor.addEventListener('input', () => {
    state.dismissedFor = null;
    // Hide stale ghost text immediately so it never sits next to text it was not computed for.
    renderGhost();
    if (!state.composing) schedulePrediction();
  });

  editor.addEventListener('keydown', (e) => {
    if (e.key === 'Tab' && !e.shiftKey && !e.ctrlKey && !e.altKey && !e.metaKey && !e.isComposing) {
      const top = state.predictions[0];
      if (top && state.forText !== null && atEnd() && caretInfo().before === state.forText
          && state.dismissedFor !== state.forText) {
        e.preventDefault();                                  // keep focus in the editor
        acceptPrediction(top);
      }                                                      // otherwise Tab keeps its normal focus behaviour
    } else if (e.key === 'Escape' && state.predictions.length) {
      state.dismissedFor = state.forText;
      renderGhost();
    }
  });

  // caret moved without typing (arrow keys, click): ghost only makes sense at the end of the text
  ['keyup', 'mouseup', 'focus'].forEach((evt) => editor.addEventListener(evt, (e) => {
    if (e.type === 'keyup' && (e.key === 'Tab' || e.key === 'Escape' || e.key === 'Shift')) return;
    renderGhost();
    if (!state.composing && caretInfo().before !== state.forText) schedulePrediction();
  }));

  editor.addEventListener('compositionstart', () => { state.composing = true; });
  editor.addEventListener('compositionend', () => { state.composing = false; renderGhost(); schedulePrediction(); });

  topkSel.addEventListener('change', () => { state.forText = null; cancelInFlight(); requestPrediction(); });

  /* model switch: no page reload, predictions refresh immediately */
  function selectModel(name, fromUser) {
    state.model = name;
    modelButtons.forEach((b) => b.setAttribute('aria-checked', String(b.dataset.model === name)));
    try { localStorage.setItem('nw-model', name); } catch (_) { /* storage may be unavailable */ }
    if (fromUser) {
      cancelInFlight();
      state.predictions = [];
      state.forText = null;
      renderChips();
      renderGhost();
      requestPrediction();
      editor.focus();
    }
  }

  modelButtons.forEach((b) => b.addEventListener('click', () => { if (b.dataset.model !== state.model) selectModel(b.dataset.model, true); }));
  document.getElementById('model-switch').addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
    const i = modelButtons.findIndex((b) => b.dataset.model === state.model);
    const next = modelButtons[(i + (e.key === 'ArrowRight' ? 1 : -1) + modelButtons.length) % modelButtons.length];
    selectModel(next.dataset.model, true);
    next.focus();
  });

  /* ---------------------------------------------------------------- init */
  let initial = cfg.models.includes('LSTM') ? 'LSTM' : (cfg.models[0] || 'LSTM');
  try {
    const saved = localStorage.getItem('nw-model');
    if (saved && cfg.models.includes(saved)) initial = saved;
  } catch (_) { /* ignore */ }
  selectModel(initial, false);
  renderGhost();
  if (cfg.ready) {
    editor.focus();
    if (editor.value.trim()) schedulePrediction(0);   // restored text after a soft reload
  }
})();
