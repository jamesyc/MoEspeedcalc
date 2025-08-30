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
  html += `<tr><td>MoE active param count</td><td>${fmt(r.moeLayers * r.expertsActivePerLayer)}</td></tr>`; // C × T × (min(J, I) ÷ I) or C × T × min(J, I)
  html += `<tr><td>MoE share of active (%)</td><td>${r.moeExpertsPct.toFixed(4)}%</td></tr>`; // AN
  html += `<tr><td>Total MoE param count (excluding shared expert)</td><td>${fmt(r.moeExpertTotal)}</td></tr>`; // C × T × I (experts total across MoE layers)
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
  const acLetters = r.hasShared ? (r.sharedScope === 'per_layer' ? 'N + O + S + M = AC' : 'N + O + S + M/C = AC') : 'N + O + S = AC';
  html += renderRow('AC', 'MoE layers always-active per-layer params', fmt(r.mAlwaysPerLayer), acNumeric, acLetters);

  const expertsLayerExplainNum = r.expertsIncludeDim ? `${fmt(r.mExpertsInput)} = ${fmt(r.expertsPerLayerTotal)}` : `${fmt(r.mExpertsInput)} × ${fmt(r.expertsPer)} = ${fmt(r.expertsPerLayerTotal)}`;
  const expertsLayerExplainLetters = r.expertsIncludeDim ? 'T = AD' : 'T × I = AD';
  html += renderRow('AD', 'MoE experts per-layer params', fmt(r.expertsPerLayerTotal), expertsLayerExplainNum, expertsLayerExplainLetters);

  const moeTotalNum = `${fmt(r.moeLayers)} × (${fmt(r.mAlwaysPerLayer)} + ${fmt(r.expertsPerLayerTotal)}) = ${fmt(r.moeTotal)}`;
  const moeTotalLetters = 'C × (AC + AD) = AE';
  html += renderRow('AE', 'MoE layers total params', fmt(r.moeTotal), moeTotalNum, moeTotalLetters);

  const expertsActiveExplainNum = r.expertsIncludeDim ? (r.expertsPer > 0 ? `${fmt(r.mExpertsInput)} × (${fmt(r.activeExpertsClamped)} ÷ ${fmt(r.expertsPer)})` : `${fmt(r.mExpertsInput)} × 0`) : `${fmt(r.mExpertsInput)} × ${fmt(r.activeExpertsClamped)}`;
  const expertsActiveExplainLetters = r.expertsIncludeDim ? (r.expertsPer > 0 ? 'T × (min(J, I) ÷ I)' : 'T × 0') : 'T × min(J, I)';
  const moeActiveNum = `${fmt(r.moeLayers)} × (${fmt(r.mAlwaysPerLayer)} + ${expertsActiveExplainNum}) = ${fmt(r.moeActive)}`;
  const moeActiveLetters = `C × (AC + ${expertsActiveExplainLetters}) = AF`;
  html += renderRow('AF', 'MoE layers total active params', fmt(r.moeActive), moeActiveNum, moeActiveLetters);

  const moeAlwaysTotalNum = `${fmt(r.moeLayers)} × ${fmt(r.mAlwaysPerLayer)} = ${fmt(r.moeAlwaysTotal)}`;
  html += renderRow('AG', 'MoE layers total always-active params', fmt(r.moeAlwaysTotal), moeAlwaysTotalNum, 'C × AC = AG');

  const moeInactiveNum = r.expertsIncludeDim
    ? `${fmt(r.moeLayers)} × ${fmt(r.mExpertsInput)} × (1 − (${fmt(r.activeExpertsClamped)} ÷ ${fmt(r.expertsPer)})) = ${fmt(r.moeInactivePerToken)}`
    : `${fmt(r.moeLayers)} × ${fmt(r.mExpertsInput)} × (${fmt(r.expertsPer)} − ${fmt(r.activeExpertsClamped)}) = ${fmt(r.moeInactivePerToken)}`;
  const moeInactiveLetters = r.expertsIncludeDim
    ? `C × T × (1 − (min(J, I) ÷ I)) = AH`
    : `C × T × (I − min(J, I)) = AH`;
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
  html += renderRow('AO', 'Total MLP param count', fmt(r.totalMlp), totalMlpNum, `B × H + C × S + C × AD${sharedTotalLetters2} = AO`);

  const totalAttnNum = `${fmt(r.denseLayers)} × ${fmt(r.dAttn)} + ${fmt(r.moeLayers)} × ${fmt(r.mAttn)} = ${fmt(r.totalAttn)}`;
  html += renderRow('AP', 'Total attention param count', fmt(r.totalAttn), totalAttnNum, 'B × G + C × N = AP');

  // Additional explanations for summary-only rows
  // AQ: MoE experts active param count (experts only)
  const moeExpertsActiveNum = r.expertsIncludeDim
    ? `${fmt(r.moeLayers)} × ${fmt(r.mExpertsInput)} × (${fmt(r.activeExpertsClamped)} ÷ ${fmt(r.expertsPer)}) = ${fmt(r.moeLayers * r.expertsActivePerLayer)}`
    : `${fmt(r.moeLayers)} × ${fmt(r.mExpertsInput)} × ${fmt(r.activeExpertsClamped)} = ${fmt(r.moeLayers * r.expertsActivePerLayer)}`;
  const moeExpertsActiveLetters = r.expertsIncludeDim
    ? 'C × T × (min(J, I) ÷ I) = AQ'
    : 'C × T × min(J, I) = AQ';
  html += renderRow('AQ', 'MoE experts active param count', fmt(r.moeLayers * r.expertsActivePerLayer), moeExpertsActiveNum, moeExpertsActiveLetters);

  // AR: MoE experts total param count (experts only)
  const moeExpertsTotalNum = r.expertsIncludeDim
    ? `${fmt(r.moeLayers)} × ${fmt(r.mExpertsInput)} = ${fmt(r.moeExpertTotal)}`
    : `${fmt(r.moeLayers)} × ${fmt(r.mExpertsInput)} × ${fmt(r.expertsPer)} = ${fmt(r.moeExpertTotal)}`;
  const moeExpertsTotalLetters = r.expertsIncludeDim
    ? 'C × T = AR'
    : 'C × T × I = AR';
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
});
