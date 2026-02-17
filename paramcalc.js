// Param Calculator: separated logic + rendering

// Formatting helpers
function fmt(n) { return (n || 0).toLocaleString('en-US'); }

function normalizeNumStr(s) {
  return String(s)
    .replace(/[\u00A0\u202F\u2009\s_]/g, '')
    .replace(/[,']/g, '');
}

function parseShapeGroup(group) {
  const content = group.replace(/^[^\[]*\[/, '').replace(/\].*$/, '');
  const dims = content.split(/\s*,\s*/).filter(Boolean).map(d => {
    const t = normalizeNumStr(d);
    const v = t ? parseInt(t, 10) : NaN;
    return Number.isFinite(v) ? v : 0;
  });
  if (dims.length === 0) return 0;
  return dims.reduce((a, b) => a * b, 1);
}

function sumShapes(text) {
  if (!text) return 0;
  let total = 0;
  const lines = String(text).split(/\r?\n/);
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;
    const groups = line.match(/\[[^\]]*\]/g);
    if (groups) {
      for (const g of groups) total += parseShapeGroup(g);
    } else {
      total += parseShapeGroup('[' + line + ']');
    }
  }
  return total;
}

// Core computation (pure)
function computeResults(input) {
  // Integers with safe defaults
  const denseLayers = parseInt(input.dense_layers || '0', 10) || 0;
  const moeLayers = parseInt(input.moe_layers || '0', 10) || 0;
  const expertsPer = parseInt(input.experts_per_layer || '0', 10) || 0;
  const activeExperts = parseInt(input.active_experts || '0', 10) || 0;
  const expertsIncludeDim = !!input.experts_include_dim;
  const hasShared = !!input.has_shared_expert;
  const sharedScope = input.shared_expert_scope || 'per_layer';

  // Shapes → counts
  const embedCount = sumShapes(input.embedding_shapes);
  const preFirstCount = sumShapes(input.pre_first_norms);

  const dNorms = sumShapes(input.dense_norms);
  const dAttn = sumShapes(input.dense_attn);
  const dFfn = sumShapes(input.dense_ffn);
  const dPerLayer = dNorms + dAttn + dFfn;

  const mAttn = sumShapes(input.moe_attn);
  const mNormsTrans = sumShapes(input.moe_transitional);
  const mSharedFfn = sumShapes(input.moe_shared_ffn);
  const mExpertsInput = sumShapes(input.moe_experts);

  const sharedExpertParams = hasShared ? sumShapes(input.shared_expert_tensors) : 0;
  const sharedPerLayer = hasShared
    ? (sharedScope === 'per_layer' ? sharedExpertParams : (moeLayers > 0 ? (sharedExpertParams / moeLayers) : 0))
    : 0;

  const mAlwaysPerLayer = mNormsTrans + mAttn + mSharedFfn + sharedPerLayer;

  const denseTotal = denseLayers * dPerLayer;
  const expertsPerLayerTotal = expertsIncludeDim ? mExpertsInput : (mExpertsInput * expertsPer);
  const moeExpertTotal = moeLayers * expertsPerLayerTotal;
  const sharedExpertTotal = hasShared ? (sharedScope === 'per_layer' ? moeLayers * sharedExpertParams : sharedExpertParams) : 0;
  const moeTotal = moeLayers * mAlwaysPerLayer + moeExpertTotal;
  const totalParams = embedCount + preFirstCount + denseTotal + moeTotal;

  const denseActive = embedCount + preFirstCount + denseTotal;
  const activeExpertsClamped = Math.max(0, Math.min(activeExperts, expertsPer));
  const expertsActivePerLayer = expertsIncludeDim
    ? (expertsPer > 0 ? mExpertsInput * (activeExpertsClamped / expertsPer) : 0)
    : (mExpertsInput * activeExpertsClamped);
  const moeActive = moeLayers * (mAlwaysPerLayer + expertsActivePerLayer);
  const totalActive = denseActive + moeActive;

  const denseActivePct = totalActive > 0 ? (100 * denseActive / totalActive) : 0;
  const moeActivePct = totalActive > 0 ? (100 * moeActive / totalActive) : 0;
  const moeInactivePerToken = moeLayers * Math.max(0, (expertsPerLayerTotal - expertsActivePerLayer));
  const totalMlp = denseLayers * dFfn + moeLayers * (mSharedFfn + expertsPerLayerTotal) + sharedExpertTotal;
  const totalAttn = denseLayers * dAttn + moeLayers * mAttn;

  const moeAlwaysTotal = moeLayers * mAlwaysPerLayer;
  const totalAlwaysActive = denseActive + moeAlwaysTotal;
  const alwaysActivePct = totalActive > 0 ? (100 * totalAlwaysActive / totalActive) : 0;
  const moeExpertsOnly = Math.max(0, moeActive - moeAlwaysTotal);
  const moeExpertsPct = totalActive > 0 ? (100 * moeExpertsOnly / totalActive) : 0;

  return {
    // inputs
    denseLayers, moeLayers, expertsPer, activeExperts, expertsIncludeDim, hasShared, sharedScope,
    // per-section counts
    embedCount, preFirstCount, dNorms, dAttn, dFfn, dPerLayer,
    mAttn, mNormsTrans, mSharedFfn, mExpertsInput,
    sharedExpertParams, sharedPerLayer, mAlwaysPerLayer,
    // totals
    denseTotal, expertsPerLayerTotal, moeExpertTotal, sharedExpertTotal, moeTotal, totalParams,
    denseActive, activeExpertsClamped, expertsActivePerLayer, moeActive, totalActive,
    denseActivePct, moeActivePct, moeInactivePerToken, totalMlp, totalAttn,
    moeAlwaysTotal, totalAlwaysActive, alwaysActivePct, moeExpertsOnly, moeExpertsPct,
    // derived display
    totalLayersComputed: denseLayers + moeLayers,
  };
}

// Rendering helpers
function renderRow(code, title, valueStr, numericEq, lettersEq) {
  return (
    '<div class="result-row">' +
      '<div class="row-line">' +
        `<div class="left"><span class="result-title">${code}: ${title}</span></div>` +
        `<div class="right"><div class="equation">${lettersEq || ''}</div></div>` +
      '</div>' +
      '<div class="row-line narrow">' +
        `<div class="left"><div class="result-value">${valueStr}</div></div>` +
        `<div class="right"><div class="equation">${numericEq}</div></div>` +
      '</div>' +
    '</div>'
  );
}

function renderSummary(r) {
  let html = '';
  html += '<h2>Results</h2>';
  html += '<table class="results-table">';
  html += '<tbody>';
  html += `<tr><td>Exact total param count</td><td>${fmt(r.totalParams)}</td></tr>`; // AJ
  html += `<tr><td>Exact active param count</td><td>${fmt(r.totalActive)}</td></tr>`; // AK
  html += `<tr><td>Total always-active param count</td><td>${fmt(r.totalAlwaysActive)}</td></tr>`; // AL
  html += `<tr><td>Always-active share of active (%)</td><td>${r.alwaysActivePct.toFixed(4)}%</td></tr>`; // AM
  html += `<tr><td>MoE active param count</td><td>${fmt(r.moeLayers * r.expertsActivePerLayer)}</td></tr>`; // C × Q × (min(J, I) ÷ I) or C × Q × min(J, I)
  html += `<tr><td>MoE share of active (%)</td><td>${r.moeExpertsPct.toFixed(4)}%</td></tr>`; // AN
  html += `<tr><td>Total MoE param count</td><td>${fmt(r.moeExpertTotal)}</td></tr>`; // C × Q × I (experts total across MoE layers)
  html += `<tr><td>Total MLP param count</td><td>${fmt(r.totalMlp)}</td></tr>`; // AO
  html += `<tr><td>MoE inactive per token count</td><td>${fmt(r.moeInactivePerToken)}</td></tr>`; // AH
  html += `<tr><td>Total attention param count</td><td>${fmt(r.totalAttn)}</td></tr>`; // AP
  html += '</tbody>';
  html += '</table>';
  return html;
}

function renderExplanation(r) {
  let html = '';
  html += '<h2>Explanation</h2>';

  html += renderRow('AA', 'Dense layer(s) per-layer params', fmt(r.dPerLayer), `${fmt(r.dNorms)} + ${fmt(r.dAttn)} + ${fmt(r.dFfn)} = ${fmt(r.dPerLayer)}`, 'F + G + H = AA');

  html += renderRow('AB', 'Dense layer(s) total params', fmt(r.denseTotal), `${fmt(r.dPerLayer)} × ${fmt(r.denseLayers)} = ${fmt(r.denseTotal)}`, 'AA × B = AB');

  const acNumeric = `${fmt(r.mAttn)} + ${fmt(r.mNormsTrans)} + ${fmt(r.mSharedFfn)} + ${fmt(r.sharedPerLayer)} = ${fmt(r.mAlwaysPerLayer)}`;
  const acLetters = r.hasShared ? (r.sharedScope === 'per_layer' ? 'N + O + P + M = AC' : 'N + O + P + M/C = AC') : 'N + O + P = AC';
  html += renderRow('AC', 'MoE layers always-active per-layer params', fmt(r.mAlwaysPerLayer), acNumeric, acLetters);

  const expertsLayerExplainNum = r.expertsIncludeDim ? `${fmt(r.mExpertsInput)} = ${fmt(r.expertsPerLayerTotal)}` : `${fmt(r.mExpertsInput)} × ${fmt(r.expertsPer)} = ${fmt(r.expertsPerLayerTotal)}`;
  const expertsLayerExplainLetters = r.expertsIncludeDim ? 'Q = AD' : 'Q × I = AD';
  html += renderRow('AD', 'MoE experts per-layer params', fmt(r.expertsPerLayerTotal), expertsLayerExplainNum, expertsLayerExplainLetters);

  const moeTotalNum = `${fmt(r.moeLayers)} × (${fmt(r.mAlwaysPerLayer)} + ${fmt(r.expertsPerLayerTotal)}) = ${fmt(r.moeTotal)}`;
  const moeTotalLetters = 'C × (AC + AD) = AE';
  html += renderRow('AE', 'MoE layers total params', fmt(r.moeTotal), moeTotalNum, moeTotalLetters);

  const expertsActiveExplainNum = r.expertsIncludeDim ? (r.expertsPer > 0 ? `${fmt(r.mExpertsInput)} × (${fmt(r.activeExpertsClamped)} ÷ ${fmt(r.expertsPer)})` : `${fmt(r.mExpertsInput)} × 0`) : `${fmt(r.mExpertsInput)} × ${fmt(r.activeExpertsClamped)}`;
  const expertsActiveExplainLetters = r.expertsIncludeDim ? (r.expertsPer > 0 ? 'Q × (min(J, I) ÷ I)' : 'Q × 0') : 'Q × min(J, I)';
  const moeActiveNum = `${fmt(r.moeLayers)} × (${fmt(r.mAlwaysPerLayer)} + ${expertsActiveExplainNum}) = ${fmt(r.moeActive)}`;
  const moeActiveLetters = `C × (AC + ${expertsActiveExplainLetters}) = AF`;
  html += renderRow('AF', 'MoE layers total active params', fmt(r.moeActive), moeActiveNum, moeActiveLetters);

  const moeAlwaysTotalNum = `${fmt(r.moeLayers)} × ${fmt(r.mAlwaysPerLayer)} = ${fmt(r.moeAlwaysTotal)}`;
  html += renderRow('AG', 'MoE layers total always-active params', fmt(r.moeAlwaysTotal), moeAlwaysTotalNum, 'C × AC = AG');

  const moeInactiveNum = r.expertsIncludeDim
    ? `${fmt(r.moeLayers)} × ${fmt(r.mExpertsInput)} × (1 − (${fmt(r.activeExpertsClamped)} ÷ ${fmt(r.expertsPer)})) = ${fmt(r.moeInactivePerToken)}`
    : `${fmt(r.moeLayers)} × ${fmt(r.mExpertsInput)} × (${fmt(r.expertsPer)} − ${fmt(r.activeExpertsClamped)}) = ${fmt(r.moeInactivePerToken)}`;
  const moeInactiveLetters = r.expertsIncludeDim
    ? `C × Q × (1 − (min(J, I) ÷ I)) = AH`
    : `C × Q × (I − min(J, I)) = AH`;
  html += renderRow('AH', 'MoE inactive per token param count', fmt(r.moeInactivePerToken), moeInactiveNum, moeInactiveLetters);

  const denseActiveNum = `${fmt(r.embedCount)} + ${fmt(r.preFirstCount)} + ${fmt(r.denseTotal)} = ${fmt(r.denseActive)}`;
  html += renderRow('AI', 'Dense layer(s) active param count', fmt(r.denseActive), denseActiveNum, 'D + E + AB = AI');

  const totalParamsNum = `${fmt(r.denseActive)} + ${fmt(r.moeTotal)} = ${fmt(r.totalParams)}`;
  html += renderRow('AJ', 'Exact total param count', fmt(r.totalParams), totalParamsNum, 'AI + AE = AJ');

  const totalActiveNum = `${fmt(r.denseActive)} + ${fmt(r.moeActive)} = ${fmt(r.totalActive)}`;
  html += renderRow('AK', 'Total active param count', fmt(r.totalActive), totalActiveNum, 'AI + AF = AK');

  const totalAlwaysActiveNum = `${fmt(r.denseActive)} + ${fmt(r.moeAlwaysTotal)} = ${fmt(r.totalAlwaysActive)}`;
  html += renderRow('AL', 'Total always-active param count', fmt(r.totalAlwaysActive), totalAlwaysActiveNum, 'AI + AG = AL');

  const alwaysShareNum = `${fmt(r.totalAlwaysActive)} ÷ ${fmt(r.totalActive)} × 100 = ${r.alwaysActivePct.toFixed(4)}%`;
  html += renderRow('AM', 'Always-active share of active (%)', `${r.alwaysActivePct.toFixed(4)}`, alwaysShareNum, 'AL ÷ AK × 100 = AM');

  const moeExpertsShareNum = `(${fmt(r.moeActive)} − ${fmt(r.moeAlwaysTotal)}) ÷ ${fmt(r.totalActive)} × 100 = ${r.moeExpertsPct.toFixed(4)}%`;
  html += renderRow('AN', 'MoE share of active (%)', `${r.moeExpertsPct.toFixed(4)}`, moeExpertsShareNum, '(AF − AG) ÷ AK × 100 = AN');

  const sharedTotalLetters2 = r.hasShared ? (r.sharedScope === 'per_layer' ? ' + C × M' : ' + M') : '';
  const totalMlpNum = `${fmt(r.denseLayers)} × ${fmt(r.dFfn)} + ${fmt(r.moeLayers)} × ${fmt(r.mSharedFfn)} + ${fmt(r.moeLayers)} × ${fmt(r.expertsPerLayerTotal)}` + (r.hasShared ? ` + ${fmt(r.sharedExpertTotal)}` : '') + ` = ${fmt(r.totalMlp)}`;
  html += renderRow('AO', 'Total MLP param count', fmt(r.totalMlp), totalMlpNum, `B × H + C × P + C × AD${sharedTotalLetters2} = AO`);

  const totalAttnNum = `${fmt(r.denseLayers)} × ${fmt(r.dAttn)} + ${fmt(r.moeLayers)} × ${fmt(r.mAttn)} = ${fmt(r.totalAttn)}`;
  html += renderRow('AP', 'Total attention param count', fmt(r.totalAttn), totalAttnNum, 'B × G + C × N = AP');

  // Additional explanations for summary-only rows
  // AQ: MoE experts active param count (experts only)
  const moeExpertsActiveNum = r.expertsIncludeDim
    ? `${fmt(r.moeLayers)} × ${fmt(r.mExpertsInput)} × (${fmt(r.activeExpertsClamped)} ÷ ${fmt(r.expertsPer)}) = ${fmt(r.moeLayers * r.expertsActivePerLayer)}`
    : `${fmt(r.moeLayers)} × ${fmt(r.mExpertsInput)} × ${fmt(r.activeExpertsClamped)} = ${fmt(r.moeLayers * r.expertsActivePerLayer)}`;
  const moeExpertsActiveLetters = r.expertsIncludeDim
    ? 'C × Q × (min(J, I) ÷ I) = AQ'
    : 'C × Q × min(J, I) = AQ';
  html += renderRow('AQ', 'MoE experts active param count', fmt(r.moeLayers * r.expertsActivePerLayer), moeExpertsActiveNum, moeExpertsActiveLetters);

  // AR: MoE experts total param count (experts only)
  const moeExpertsTotalNum = r.expertsIncludeDim
    ? `${fmt(r.moeLayers)} × ${fmt(r.mExpertsInput)} = ${fmt(r.moeExpertTotal)}`
    : `${fmt(r.moeLayers)} × ${fmt(r.mExpertsInput)} × ${fmt(r.expertsPer)} = ${fmt(r.moeExpertTotal)}`;
  const moeExpertsTotalLetters = r.expertsIncludeDim
    ? 'C × Q = AR'
    : 'C × Q × I = AR';
  html += renderRow('AR', 'MoE experts total param count', fmt(r.moeExpertTotal), moeExpertsTotalNum, moeExpertsTotalLetters);

  return html;
}

// DOM wiring
function getFormData() {
  return {
    total_layers: document.getElementById('total_layers')?.value || '0',
    dense_layers: document.getElementById('dense_layers')?.value || '0',
    moe_layers: document.getElementById('moe_layers')?.value || '0',
    embedding_shapes: document.getElementById('embedding_shapes')?.value || '',
    pre_first_norms: document.getElementById('pre_first_norms')?.value || '',
    dense_norms: document.getElementById('dense_norms')?.value || '',
    dense_attn: document.getElementById('dense_attn')?.value || '',
    dense_ffn: document.getElementById('dense_ffn')?.value || '',
    experts_per_layer: document.getElementById('experts_per_layer')?.value || '0',
    active_experts: document.getElementById('active_experts')?.value || '0',
    has_shared_expert: document.getElementById('has_shared_expert')?.checked || false,
    shared_expert_scope: document.getElementById('shared_expert_scope')?.value || 'per_layer',
    shared_expert_tensors: document.getElementById('shared_expert_tensors')?.value || '',
    moe_attn: document.getElementById('moe_attn')?.value || '',
    moe_transitional: document.getElementById('moe_transitional')?.value || '',
    moe_shared_ffn: document.getElementById('moe_shared_ffn')?.value || '',
    moe_experts: document.getElementById('moe_experts')?.value || '',
    experts_include_dim: document.getElementById('experts_include_dim')?.checked || false,
  };
}

function updateComputedTotalLayers() {
  const b = parseInt(document.getElementById('dense_layers')?.value || '0', 10) || 0;
  const c = parseInt(document.getElementById('moe_layers')?.value || '0', 10) || 0;
  const aEl = document.getElementById('total_layers');
  if (aEl) aEl.value = String(b + c);
}

function calculateAndRender(opts = {}) {
  const { scroll = false } = opts;
  const resultsBox = document.getElementById('results');
  const explanationBox = document.getElementById('explanation');
  const input = getFormData();
  const r = computeResults(input);

  // Update computed total layers (A = B + C)
  updateComputedTotalLayers();

  resultsBox.innerHTML = renderSummary(r);
  explanationBox.innerHTML = renderExplanation(r);
  resultsBox.classList.remove('hidden');
  explanationBox.classList.remove('hidden');
  if (scroll && typeof resultsBox.scrollIntoView === 'function') {
    resultsBox.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

function debounce(fn, ms) {
  let t = null;
  return function(...args) {
    clearTimeout(t);
    t = setTimeout(() => fn.apply(this, args), ms);
  };
}

function updateSharedState() {
  const chk = document.getElementById('has_shared_expert');
  const scopeSel = document.getElementById('shared_expert_scope');
  const sharedTA = document.getElementById('shared_expert_tensors');
  const en = !!(chk && chk.checked);
  if (scopeSel) scopeSel.disabled = !en;
  if (sharedTA) sharedTA.disabled = !en;
}

// Prefill presets for the parameter calculator
function prefillParamModel(model) {
  const setVal = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };
  const setChecked = (id, v) => { const el = document.getElementById(id); if (el) el.checked = v; };

  if (model === 'kimi-k2') {
    // B, C
    setVal('dense_layers', '1'); // B
    setVal('moe_layers', '60'); // C

    // D: Embedding/output matrices
    setVal('embedding_shapes', [
      '[163840, 7168]',
      '[163840, 7168]'
    ].join('\n'));

    // E: Pre/post first/last norms (optional)
    setVal('pre_first_norms', [
      '[7168]'
    ].join('\n'));

    // F: Dense layer norms
    setVal('dense_norms', [
      '[7168]',
      '[7168]'
    ].join('\n'));

    // G: Dense attention tensors
    setVal('dense_attn', [
      '[512]',
      '[576, 7168]',
      '[5, 56]',
      '[16384, 512]',
      '[128, 4]',
      '[7168, 8192]',
      '[56, 64]',
      '[1536]',
      '[1536, 7168]',
      '[12, 56]',
      '[12288, 1536]',
      '[96, 12]',
      '[56]'
    ].join('\n'));

    // H: Dense FFN tensors
    setVal('dense_ffn', [
      '[7168, 18432]',
      '[56, 144]',
      '[18432, 7168]',
      '[144, 56]',
      '[18432, 7168]',
      '[144, 56]'
    ].join('\n'));

    // I, J
    setVal('experts_per_layer', '384'); // I
    setVal('active_experts', '8'); // J

    // K, L, M: Shared expert
    setChecked('has_shared_expert', true);
    updateSharedState();
    setVal('shared_expert_scope', 'per_layer');
    setVal('shared_expert_tensors', [
      '[7168, 2048]',
      '[56, 16]',
      '[7168, 2048]',
      '[56, 16]',
      '[7168, 2048]',
      '[56, 16]'
    ].join('\n'));

    // N: MoE attention tensors (same set as G per description)
    setVal('moe_attn', [
      '[512]',
      '[576, 7168]',
      '[5, 56]',
      '[16384, 512]',
      '[128, 4]',
      '[7168, 8192]',
      '[56, 64]',
      '[1536]',
      '[1536, 7168]',
      '[12, 56]',
      '[12288, 1536]',
      '[96, 12]',
      '[56]'
    ].join('\n'));

    // O: MoE norms/transitional
    setVal('moe_transitional', [
      '[7168]',
      '[7168]'
    ].join('\n'));

    // P: Shared FFN (always active)
    setVal('moe_shared_ffn', [
      '[384]',
      '[384, 7168]'
    ].join('\n'));

    // Q: MoE experts tensors (per layer; may include E)
    setVal('moe_experts', [
      '[7168, 2048]',
      '[56, 16]',
      '[2048, 7168]',
      '[16, 56]',
      '[2048, 7168]',
      '[16, 56]'
    ].join('\n'));

    // Experts include E: unchecked per request
    setChecked('experts_include_dim', false);

    // Update computed A and render
    updateComputedTotalLayers();
    calculateAndRender({ scroll: false });
  } else if (model === 'glm-4.7') {
    // B, C
    setVal('dense_layers', '3'); // B
    setVal('moe_layers', '89'); // C

    // D: Embedding/output matrices
    setVal('embedding_shapes', [
      '[151552, 5120]',
      '[151552, 5120]'
    ].join('\n'));

    // E: Pre/post first/last norms (optional)
    setVal('pre_first_norms', [
      '[5120]'
    ].join('\n'));

    // F: Dense layer norms
    setVal('dense_norms', [
      '[5120]',
      '[5120]',
      '[128]',
      '[128]'
    ].join('\n'));

    // G: Dense attention tensors
    setVal('dense_attn', [
      '[12288, 5120]',
      '[12288]',
      '[1024, 5120]',
      '[1024]',
      '[1024, 5120]',
      '[1024]',
      '[5120, 12288]'
    ].join('\n'));

    // H: Dense FFN tensors
    setVal('dense_ffn', [
      '[12288, 5120]',
      '[12288, 5120]',
      '[5120, 12288]'
    ].join('\n'));

    // I, J
    setVal('experts_per_layer', '160'); // I
    setVal('active_experts', '8'); // J

    // K, L, M: Shared expert
    setChecked('has_shared_expert', true);
    updateSharedState();
    setVal('shared_expert_scope', 'per_layer');
    setVal('shared_expert_tensors', [
      '[1536, 5120]',
      '[1536, 5120]',
      '[5120, 1536]'
    ].join('\n'));

    // N: MoE attention tensors (same set as G)
    setVal('moe_attn', [
      '[12288, 5120]',
      '[12288]',
      '[1024, 5120]',
      '[1024]',
      '[1024, 5120]',
      '[1024]',
      '[5120, 12288]'
    ].join('\n'));

    // O: MoE norms/transitional
    setVal('moe_transitional', [
      '[5120]',
      '[5120]',
      '[128]',
      '[128]'
    ].join('\n'));

    // P: Gate input, biases, other FFN (always active)
    setVal('moe_shared_ffn', [
      '[160, 5120]',
      '[160]'
    ].join('\n'));

    // Q: MoE experts tensors (includes expert dimension E)
    setVal('moe_experts', [
      '[160, 1536, 5120]',
      '[160, 1536, 5120]',
      '[160, 5120, 1536]'
    ].join('\n'));

    // Experts include E: checked
    setChecked('experts_include_dim', true);

    // Update computed A and render
    updateComputedTotalLayers();
    calculateAndRender({ scroll: false });
  } else if (model === 'glm-5') {
    // B, C
    setVal('dense_layers', '3'); // B
    setVal('moe_layers', '75'); // C

    // D: Embedding/output matrices
    // GLM stores lm_head as [V, D]; calculator expects output projection as [D, V]
    setVal('embedding_shapes', [
      '[154880, 6144]',
      '[6144, 154880]'
    ].join('\n'));

    // E: Pre/post first/last norms (optional)
    setVal('pre_first_norms', [
      '[6144]'
    ].join('\n'));

    // F: Dense layer norms
    setVal('dense_norms', [
      '[6144]',
      '[6144]',
      '[2048]',
      '[512]',
      '[128]',
      '[128]'
    ].join('\n'));

    // G: Dense attention tensors
    setVal('dense_attn', [
      '[2048, 6144]',
      '[16384, 2048]',
      '[576, 6144]',
      '[28672, 512]',
      '[6144, 16384]',
      '[32, 6144]',
      '[128, 6144]',
      '[4096, 2048]'
    ].join('\n'));

    // H: Dense FFN tensors
    setVal('dense_ffn', [
      '[12288, 6144]',
      '[12288, 6144]',
      '[6144, 12288]'
    ].join('\n'));

    // I, J
    setVal('experts_per_layer', '256'); // I
    setVal('active_experts', '8'); // J

    // K, L, M: Shared expert
    setChecked('has_shared_expert', true);
    updateSharedState();
    setVal('shared_expert_scope', 'per_layer');
    setVal('shared_expert_tensors', [
      '[2048, 6144]',
      '[2048, 6144]',
      '[6144, 2048]'
    ].join('\n'));

    // N: MoE attention tensors (same set as G)
    setVal('moe_attn', [
      '[2048, 6144]',
      '[16384, 2048]',
      '[576, 6144]',
      '[28672, 512]',
      '[6144, 16384]',
      '[32, 6144]',
      '[128, 6144]',
      '[4096, 2048]'
    ].join('\n'));

    // O: MoE norms/transitional
    setVal('moe_transitional', [
      '[6144]',
      '[6144]',
      '[2048]',
      '[512]',
      '[128]',
      '[128]'
    ].join('\n'));

    // P: Gate input, biases, other FFN (always active)
    setVal('moe_shared_ffn', [
      '[256, 6144]',
      '[256]'
    ].join('\n'));

    // Q: MoE experts tensors (includes expert dimension E)
    setVal('moe_experts', [
      '[256, 6144, 2048]',
      '[256, 2048, 6144]',
      '[256, 2048, 6144]'
    ].join('\n'));

    // Experts include E: checked
    setChecked('experts_include_dim', true);

    // Update computed A and render
    updateComputedTotalLayers();
    calculateAndRender({ scroll: false });
  } else if (model === 'deepseek-v3') {
    // B, C
    setVal('dense_layers', '3'); // B
    setVal('moe_layers', '58'); // C

    // D: Embedding/output matrices
    setVal('embedding_shapes', [
      '[129280, 7168]',
      '[129280, 7168]'
    ].join('\n'));

    // E: Pre/post first/last norms (optional)
    setVal('pre_first_norms', [
      '[7168]'
    ].join('\n'));

    // F: Dense layer norms
    setVal('dense_norms', [
      '[7168]',
      '[7168]'
    ].join('\n'));

    // G: Dense attention tensors
    setVal('dense_attn', [
      '[512]',
      '[576, 7168]',
      '[5, 56]',
      '[32768, 512]',
      '[256, 4]',
      '[7168, 16384]',
      '[56, 128]',
      '[1536]',
      '[1536, 7168]',
      '[12, 56]',
      '[24576, 1536]',
      '[192, 12]'
    ].join('\n'));

    // H: Dense FFN tensors
    setVal('dense_ffn', [
      '[7168, 18432]',
      '[56, 144]',
      '[18432, 7168]',
      '[144, 56]',
      '[18432, 7168]',
      '[144, 56]'
    ].join('\n'));

    // I, J
    setVal('experts_per_layer', '256'); // I
    setVal('active_experts', '8'); // J

    // K, L, M: Shared expert
    setChecked('has_shared_expert', true);
    updateSharedState();
    setVal('shared_expert_scope', 'per_layer');
    setVal('shared_expert_tensors', [
      '[7168, 2048]',
      '[56, 16]',
      '[7168, 2048]',
      '[56, 16]',
      '[7168, 2048]',
      '[56, 16]'
    ].join('\n'));

    // N: MoE attention tensors (same set as G)
    setVal('moe_attn', [
      '[512]',
      '[576, 7168]',
      '[5, 56]',
      '[32768, 512]',
      '[256, 4]',
      '[7168, 16384]',
      '[56, 128]',
      '[1536]',
      '[1536, 7168]',
      '[12, 56]',
      '[24576, 1536]',
      '[192, 12]'
    ].join('\n'));

    // O: MoE norms/transitional
    setVal('moe_transitional', [
      '[7168]',
      '[7168]'
    ].join('\n'));

    // P: Shared FFN (always active)
    setVal('moe_shared_ffn', [
      '[256]',
      '[256, 7168]'
    ].join('\n'));

    // Q: MoE experts tensors (per layer; may include E)
    setVal('moe_experts', [
      '[7168, 2048]',
      '[56, 16]',
      '[2048, 7168]',
      '[16, 56]',
      '[2048, 7168]',
      '[16, 56]'
    ].join('\n'));

    // Experts include E: unchecked per request
    setChecked('experts_include_dim', false);

    // Update computed A and render
    updateComputedTotalLayers();
    calculateAndRender({ scroll: false });
  } else if (model === 'qwen3-235b') {
    // B, C
    setVal('dense_layers', '0'); // B
    setVal('moe_layers', '94'); // C

    // D: Embedding/output matrices
    setVal('embedding_shapes', [
      '[151936, 4096]',
      '[151936, 4096]'
    ].join('\n'));

    // E: Pre/post first/last norms (optional)
    setVal('pre_first_norms', [
      '[4096]'
    ].join('\n'));

    // F/G/H: Dense layers are unused for this model
    setVal('dense_norms', '');
    setVal('dense_attn', '');
    setVal('dense_ffn', '');

    // I, J
    setVal('experts_per_layer', '128'); // I
    setVal('active_experts', '8'); // J

    // K, L, M: Shared expert disabled
    setChecked('has_shared_expert', false);
    updateSharedState();
    setVal('shared_expert_scope', 'per_layer');
    setVal('shared_expert_tensors', '');

    // N: MoE attention tensors
    setVal('moe_attn', [
      '[8192, 4096]',
      '[512, 4096]',
      '[512, 4096]',
      '[4096, 8192]'
    ].join('\n'));

    // O: MoE norms/transitional
    setVal('moe_transitional', [
      '[4096]',
      '[4096]',
      '[128]',
      '[128]'
    ].join('\n'));

    // P: Gate input, biases, other FFN (always active)
    setVal('moe_shared_ffn', [
      '[128, 4096]'
    ].join('\n'));

    // Q: MoE experts tensors (includes expert dimension E)
    setVal('moe_experts', [
      '[128, 1536, 4096]',
      '[128, 1536, 4096]',
      '[128, 4096, 1536]'
    ].join('\n'));

    // Experts include E: checked
    setChecked('experts_include_dim', true);

    // Update computed A and render
    updateComputedTotalLayers();
    calculateAndRender({ scroll: false });
  } else if (model === 'deepseek-v3-mtp') {
    // Start from base Deepseek V3, then override differences
    prefillParamModel('deepseek-v3');

    const setVal = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };

    // C: 59 MoE layers
    setVal('moe_layers', '59');

    // D: four lines, mixing plain and narrow‑space formats
    setVal('embedding_shapes', [
      '[129280, 7168]',
      '[129280, 7168]',
      '[129\u202F280, 7\u202F168]',
      '[129\u202F280, 7\u202F168]'
    ].join('\n'));

    // E: multiple lines with narrow spaces included
    setVal('pre_first_norms', [
      '[7168]',
      '[7\u202F168]',
      '[7\u202F168]',
      '[7\u202F168, 14\u202F336]',
      '[7\u202F168]'
    ].join('\n'));

    updateComputedTotalLayers();
    calculateAndRender({ scroll: false });
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('paramcalc-form');
  const btn = document.getElementById('calculate-btn');
  const debouncedCalc = debounce(() => calculateAndRender({ scroll: false }), 250);

  form?.addEventListener('submit', (e) => {
    e.preventDefault();
    calculateAndRender({ scroll: true });
  });
  btn?.addEventListener('click', (e) => { e.preventDefault(); calculateAndRender({ scroll: true }); });

  // Live updates on inputs
  const inputs = document.querySelectorAll('#paramcalc-form input, #paramcalc-form textarea, #paramcalc-form select');
  inputs.forEach(el => {
    el.addEventListener('input', () => { if (el.id === 'dense_layers' || el.id === 'moe_layers') updateComputedTotalLayers(); debouncedCalc(); });
    el.addEventListener('change', () => { if (el.id === 'dense_layers' || el.id === 'moe_layers') updateComputedTotalLayers(); debouncedCalc(); });
  });

  // Shared expert toggle state
  const chk = document.getElementById('has_shared_expert');
  chk?.addEventListener('change', () => { updateSharedState(); debouncedCalc(); });
  updateSharedState();

  // Initialize computed total layers and first render
  updateComputedTotalLayers();

  // Preset dropdown wiring
  const presetSel = document.getElementById('param-model-select');
  presetSel?.addEventListener('change', () => {
    const v = presetSel.value;
    if (v === 'custom') return; // keep user values
    prefillParamModel(v);
  });
});
