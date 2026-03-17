// Param Calculator: separated logic + rendering

function fmt(n) { return (n || 0).toLocaleString('en-US'); }

function normalizeNumStr(s) {
  return String(s)
    .replace(/[\u00A0\u202F\u2009\s_]/g, '')
    .replace(/[,']/g, '');
}

function parseDimToken(raw) {
  const t = normalizeNumStr(raw);
  if (!t) return { value: 0, valid: false };
  const v = Number(t);
  const valid = Number.isFinite(v) && v >= 0;
  return { value: valid ? v : 0, valid };
}

function parseShapeGroup(group) {
  const content = group.replace(/^[^\[]*\[/, '').replace(/\].*$/, '');
  const tokens = content.split(/\s*,\s*/).filter(Boolean);
  if (tokens.length === 0) return 0;
  const dims = tokens.map(parseDimToken);
  if (dims.some(d => !d.valid)) return 0;
  return dims.reduce((a, b) => a * b.value, 1);
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

function collectInvalidShapeEntries(text) {
  if (!text) return [];
  const invalidEntries = [];
  const lines = String(text).split(/\r?\n/);
  for (let i = 0; i < lines.length; i += 1) {
    const rawLine = lines[i];
    const line = rawLine.trim();
    if (!line) continue;
    const groups = line.match(/\[[^\]]*\]/g) || ['[' + line + ']'];
    for (const group of groups) {
      const content = group.replace(/^[^\[]*\[/, '').replace(/\].*$/, '');
      const tokens = content.split(/\s*,\s*/).filter(Boolean);
      const valid = tokens.length > 0 && tokens.every(token => parseDimToken(token).valid);
      if (!valid) invalidEntries.push({ line: i + 1, value: group });
    }
  }
  return invalidEntries;
}

function parseCount(value) {
  return parseInt(value || '0', 10) || 0;
}

function makeDenseBucket(label, count, normsText, attnText, ffnText) {
  const norms = sumShapes(normsText);
  const attn = sumShapes(attnText);
  const ffn = sumShapes(ffnText);
  const perLayer = norms + attn + ffn;
  return {
    label,
    count,
    norms,
    attn,
    ffn,
    perLayer,
    total: count * perLayer,
  };
}

function makeMoeBucket(label, count, attnText, transitionalText, sharedFfnText, expertsText, expertsIncludeDim, expertsPer, activeExpertsClamped, hasShared, sharedPerLayer) {
  const attn = sumShapes(attnText);
  const normsTrans = sumShapes(transitionalText);
  const sharedFfn = sumShapes(sharedFfnText);
  const expertsInput = sumShapes(expertsText);
  const bucketSharedPerLayer = count > 0 ? sharedPerLayer : 0;
  const alwaysPerLayer = attn + normsTrans + sharedFfn + bucketSharedPerLayer;
  const expertsPerLayerTotal = expertsIncludeDim ? expertsInput : (expertsInput * expertsPer);
  const expertsActivePerLayer = expertsIncludeDim
    ? (expertsPer > 0 ? expertsInput * (activeExpertsClamped / expertsPer) : 0)
    : (expertsInput * activeExpertsClamped);
  return {
    label,
    count,
    attn,
    normsTrans,
    sharedFfn,
    expertsInput,
    sharedPerLayer: bucketSharedPerLayer,
    alwaysPerLayer,
    expertsPerLayerTotal,
    expertsActivePerLayer,
    alwaysTotal: count * alwaysPerLayer,
    expertTotal: count * expertsPerLayerTotal,
    activeTotal: count * (alwaysPerLayer + expertsActivePerLayer),
    inactivePerToken: count * Math.max(0, expertsPerLayerTotal - expertsActivePerLayer),
    mlpTotal: count * (sharedFfn + expertsPerLayerTotal + (hasShared ? bucketSharedPerLayer : 0)),
  };
}

function countWarnings(input) {
  return [
    ['E', collectInvalidShapeEntries(input.embedding_shapes)],
    ['F', collectInvalidShapeEntries(input.pre_first_norms)],
    ['G', collectInvalidShapeEntries(input.dense_norms)],
    ['H', collectInvalidShapeEntries(input.dense_attn)],
    ['I', collectInvalidShapeEntries(input.dense_ffn)],
    ['J', collectInvalidShapeEntries(input.dense_ssm_norms)],
    ['K', collectInvalidShapeEntries(input.dense_ssm_attn)],
    ['L', collectInvalidShapeEntries(input.dense_ssm_ffn)],
    ['P', collectInvalidShapeEntries(input.shared_expert_tensors)],
    ['Q', collectInvalidShapeEntries(input.moe_attn)],
    ['R', collectInvalidShapeEntries(input.moe_transitional)],
    ['S', collectInvalidShapeEntries(input.moe_shared_ffn)],
    ['T', collectInvalidShapeEntries(input.moe_experts)],
    ['U', collectInvalidShapeEntries(input.moe_ssm_attn)],
    ['V', collectInvalidShapeEntries(input.moe_ssm_transitional)],
    ['W', collectInvalidShapeEntries(input.moe_ssm_shared_ffn)],
    ['X', collectInvalidShapeEntries(input.moe_ssm_experts)],
  ].filter(([, entries]) => entries.length > 0);
}

const STABLE_LABEL_REFS = Object.freeze({
  dense_attention_layers: 'Z01',
  dense_ssm_attention_layers: 'Z02',
  moe_attention_layers: 'Z03',
  moe_ssm_attention_layers: 'Z04',
  embedding_shapes: 'Z05',
  pre_first_norms: 'Z06',
  dense_norms: 'Z07',
  dense_attn: 'Z08',
  dense_ffn: 'Z09',
  dense_ssm_norms: 'Z10',
  dense_ssm_attn: 'Z11',
  dense_ssm_ffn: 'Z12',
  experts_per_layer: 'Z13',
  active_experts: 'Z14',
  shared_expert_scope: 'Z15',
  shared_expert_tensors: 'Z16',
  moe_attn: 'Z17',
  moe_transitional: 'Z18',
  moe_shared_ffn: 'Z19',
  moe_experts: 'Z20',
  moe_ssm_attn: 'Z21',
  moe_ssm_transitional: 'Z22',
  moe_ssm_shared_ffn: 'Z23',
  moe_ssm_experts: 'Z24',
  has_shared_expert: 'Z43',
  experts_include_dim: 'Z44',
  explanation_dense_attention_only: 'Z25',
  explanation_dense_ssm_attention: 'Z26',
  explanation_dense_total: 'Z27',
  explanation_moe_attention_only_always: 'Z28',
  explanation_moe_ssm_attention_always: 'Z29',
  explanation_moe_attention_only_experts: 'Z30',
  explanation_moe_ssm_attention_experts: 'Z31',
  explanation_moe_total: 'Z32',
  explanation_moe_active: 'Z33',
  explanation_moe_always_total: 'Z45',
  explanation_dense_active: 'Z34',
  explanation_total_params: 'Z35',
  explanation_total_active: 'Z36',
  explanation_total_always_active: 'Z37',
  explanation_always_share: 'Z38',
  explanation_moe_share_active: 'Z47',
  explanation_total_mlp: 'Z39',
  explanation_total_attn: 'Z40',
  explanation_moe_experts_active: 'Z41',
  explanation_moe_experts_total: 'Z42',
  explanation_moe_inactive_per_token: 'Z46',
});

const FORM_STABLE_LABEL_REFS = Object.freeze(
  Object.fromEntries(
    Object.entries(STABLE_LABEL_REFS).filter(([key]) => !key.startsWith('explanation_')),
  ),
);

const FIELD_IDS_BY_STABLE_REF = Object.freeze(
  Object.fromEntries(
    Object.entries(FORM_STABLE_LABEL_REFS).map(([fieldId, stableRef]) => [stableRef, fieldId]),
  ),
);

function createEmptyInput(overrides = {}) {
  return {
    total_layers: '0',
    dense_layers: '0',
    moe_layers: '0',
    dense_attention_layers: '0',
    dense_ssm_attention_layers: '0',
    moe_attention_layers: '0',
    moe_ssm_attention_layers: '0',
    embedding_shapes: '',
    pre_first_norms: '',
    dense_norms: '',
    dense_attn: '',
    dense_ffn: '',
    dense_ssm_norms: '',
    dense_ssm_attn: '',
    dense_ssm_ffn: '',
    experts_per_layer: '0',
    active_experts: '0',
    has_shared_expert: false,
    shared_expert_scope: 'per_layer',
    shared_expert_tensors: '',
    moe_attn: '',
    moe_transitional: '',
    moe_shared_ffn: '',
    moe_experts: '',
    moe_ssm_attn: '',
    moe_ssm_transitional: '',
    moe_ssm_shared_ffn: '',
    moe_ssm_experts: '',
    experts_include_dim: false,
    ...overrides,
  };
}

function readPresetDataFromFs() {
  if (typeof require === 'undefined') return null;
  const fs = require('node:fs');
  const path = require('node:path');
  return JSON.parse(fs.readFileSync(path.join(__dirname, 'paramcalc.presets.json'), 'utf8'));
}

let presetDataCache = null;
let presetDataPromise = null;

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

function getPresetDataFromGlobal() {
  if (typeof globalThis === 'undefined') return null;
  const data = globalThis.PARAMCALC_PRESETS;
  return data && typeof data === 'object' ? data : null;
}

async function loadPresetData() {
  if (presetDataCache) return presetDataCache;
  const globalData = getPresetDataFromGlobal();
  if (globalData) {
    presetDataCache = cloneJson(globalData);
    return presetDataCache;
  }
  if (typeof window === 'undefined') {
    presetDataCache = readPresetDataFromFs();
    return presetDataCache;
  }
  if (!presetDataPromise) {
    presetDataPromise = fetch('paramcalc.presets.json')
      .then((response) => {
        if (!response.ok) throw new Error(`Failed to load preset JSON: ${response.status}`);
        return response.json();
      })
      .then((data) => {
        presetDataCache = data;
        return data;
      });
  }
  return presetDataPromise;
}

function getPresetDataSync() {
  if (presetDataCache) return presetDataCache;
  const globalData = getPresetDataFromGlobal();
  if (globalData) {
    presetDataCache = cloneJson(globalData);
    return presetDataCache;
  }
  presetDataCache = readPresetDataFromFs();
  return presetDataCache;
}

function applyStableRefPresetData(presetEntry, setVal, setChecked) {
  Object.entries(presetEntry || {}).forEach(([stableRef, value]) => {
    const fieldId = FIELD_IDS_BY_STABLE_REF[stableRef];
    if (!fieldId) return;
    if (typeof value === 'boolean') {
      setChecked(fieldId, value);
    } else {
      setVal(fieldId, value);
    }
  });
}

async function applyPresetModel(model, setVal, setChecked) {
  const presetData = await loadPresetData();
  const presetEntry = presetData?.models?.[model];
  if (!presetEntry) return false;
  applyStableRefPresetData(presetEntry, setVal, setChecked);
  return true;
}

function buildPresetInput(model) {
  const input = createEmptyInput();
  const setVal = (id, value) => {
    input[id] = value;
  };
  const setChecked = (id, value) => {
    input[id] = !!value;
  };
  const presetData = getPresetDataSync();
  const presetEntry = presetData?.models?.[model];
  if (!presetEntry) {
    throw new Error(`Unknown preset model: ${model}`);
  }
  applyStableRefPresetData(cloneJson(presetEntry), setVal, setChecked);
  input.total_layers = String(
    parseCount(input.dense_attention_layers)
    + parseCount(input.dense_ssm_attention_layers)
    + parseCount(input.moe_attention_layers)
    + parseCount(input.moe_ssm_attention_layers),
  );
  return input;
}

function getPresetModels() {
  const presetData = getPresetDataSync();
  return [...(presetData?.modelOrder || Object.keys(presetData?.models || {}))];
}

function computeResults(input) {
  const denseAttentionLayersRaw = parseCount(input.dense_attention_layers);
  const denseSsmAttentionLayers = parseCount(input.dense_ssm_attention_layers);
  const moeAttentionLayersRaw = parseCount(input.moe_attention_layers);
  const moeSsmAttentionLayers = parseCount(input.moe_ssm_attention_layers);
  const hasExplicitDenseCounts = denseAttentionLayersRaw > 0 || denseSsmAttentionLayers > 0;
  const hasExplicitMoeCounts = moeAttentionLayersRaw > 0 || moeSsmAttentionLayers > 0;

  const denseAttentionLayers = hasExplicitDenseCounts ? denseAttentionLayersRaw : parseCount(input.dense_layers);
  const moeAttentionLayers = hasExplicitMoeCounts ? moeAttentionLayersRaw : parseCount(input.moe_layers);
  const denseLayers = denseAttentionLayers + denseSsmAttentionLayers;
  const moeLayers = moeAttentionLayers + moeSsmAttentionLayers;
  const expertsPer = parseCount(input.experts_per_layer);
  const activeExperts = parseCount(input.active_experts);
  const activeExpertsClamped = Math.max(0, Math.min(activeExperts, expertsPer));
  const expertsIncludeDim = !!input.experts_include_dim;
  const hasShared = !!input.has_shared_expert;
  const sharedScope = input.shared_expert_scope || 'per_layer';

  const embedCount = sumShapes(input.embedding_shapes);
  const preFirstCount = sumShapes(input.pre_first_norms);
  const sharedExpertParams = hasShared ? sumShapes(input.shared_expert_tensors) : 0;
  const sharedPerLayer = hasShared
    ? (sharedScope === 'per_layer' ? sharedExpertParams : (moeLayers > 0 ? (sharedExpertParams / moeLayers) : 0))
    : 0;
  const sharedExpertTotal = hasShared ? (sharedScope === 'per_layer' ? moeLayers * sharedExpertParams : sharedExpertParams) : 0;

  const denseAttentionOnly = makeDenseBucket(
    'Dense attention-only',
    denseAttentionLayers,
    input.dense_norms,
    input.dense_attn,
    input.dense_ffn,
  );
  const denseSsmAttention = makeDenseBucket(
    'Dense SSM+attention',
    denseSsmAttentionLayers,
    input.dense_ssm_norms,
    input.dense_ssm_attn,
    input.dense_ssm_ffn,
  );

  const moeAttentionOnly = makeMoeBucket(
    'MoE attention-only',
    moeAttentionLayers,
    input.moe_attn,
    input.moe_transitional,
    input.moe_shared_ffn,
    input.moe_experts,
    expertsIncludeDim,
    expertsPer,
    activeExpertsClamped,
    hasShared,
    sharedPerLayer,
  );
  const moeSsmAttention = makeMoeBucket(
    'MoE SSM+attention',
    moeSsmAttentionLayers,
    input.moe_ssm_attn,
    input.moe_ssm_transitional,
    input.moe_ssm_shared_ffn,
    input.moe_ssm_experts,
    expertsIncludeDim,
    expertsPer,
    activeExpertsClamped,
    hasShared,
    sharedPerLayer,
  );

  const dNorms = denseAttentionOnly.norms + denseSsmAttention.norms;
  const dAttn = denseAttentionOnly.attn + denseSsmAttention.attn;
  const dFfn = denseAttentionOnly.ffn + denseSsmAttention.ffn;
  const denseTotal = denseAttentionOnly.total + denseSsmAttention.total;

  const mAttn = moeAttentionOnly.attn + moeSsmAttention.attn;
  const mNormsTrans = moeAttentionOnly.normsTrans + moeSsmAttention.normsTrans;
  const mSharedFfn = moeAttentionOnly.sharedFfn + moeSsmAttention.sharedFfn;
  const mExpertsInput = moeAttentionOnly.expertsInput + moeSsmAttention.expertsInput;
  const moeExpertTotal = moeAttentionOnly.expertTotal + moeSsmAttention.expertTotal;
  const moeAlwaysTotal = moeAttentionOnly.alwaysTotal + moeSsmAttention.alwaysTotal;
  const moeTotal = moeAlwaysTotal + moeExpertTotal;

  const denseActive = embedCount + preFirstCount + denseTotal;
  const moeActive = moeAttentionOnly.activeTotal + moeSsmAttention.activeTotal;
  const totalParams = denseActive + moeTotal;
  const totalActive = denseActive + moeActive;
  const moeInactivePerToken = moeAttentionOnly.inactivePerToken + moeSsmAttention.inactivePerToken;
  const totalMlp = (denseAttentionOnly.count * denseAttentionOnly.ffn)
    + (denseSsmAttention.count * denseSsmAttention.ffn)
    + moeAttentionOnly.mlpTotal
    + moeSsmAttention.mlpTotal;
  const totalAttn = denseAttentionOnly.count * denseAttentionOnly.attn + denseSsmAttention.count * denseSsmAttention.attn + moeAttentionOnly.count * moeAttentionOnly.attn + moeSsmAttention.count * moeSsmAttention.attn;
  const totalAlwaysActive = denseActive + moeAlwaysTotal;
  const alwaysActivePct = totalActive > 0 ? (100 * totalAlwaysActive / totalActive) : 0;
  const denseActivePct = totalActive > 0 ? (100 * denseActive / totalActive) : 0;
  const moeActivePct = totalActive > 0 ? (100 * moeActive / totalActive) : 0;
  const moeExpertsOnly = Math.max(0, moeActive - moeAlwaysTotal);
  const moeExpertsPct = totalActive > 0 ? (100 * moeExpertsOnly / totalActive) : 0;
  const invalidShapeWarnings = countWarnings(input);

  return {
    denseLayers,
    moeLayers,
    denseAttentionLayers,
    denseSsmAttentionLayers,
    moeAttentionLayers,
    moeSsmAttentionLayers,
    expertsPer,
    activeExperts,
    activeExpertsClamped,
    expertsIncludeDim,
    hasShared,
    sharedScope,
    embedCount,
    preFirstCount,
    sharedExpertParams,
    sharedPerLayer,
    sharedExpertTotal,
    denseAttentionOnly,
    denseSsmAttention,
    moeAttentionOnly,
    moeSsmAttention,
    dNorms,
    dAttn,
    dFfn,
    dPerLayer: denseAttentionOnly.perLayer,
    dSsmPerLayer: denseSsmAttention.perLayer,
    mAttn,
    mNormsTrans,
    mSharedFfn,
    mExpertsInput,
    mAlwaysPerLayer: moeAttentionOnly.alwaysPerLayer,
    mSsmAlwaysPerLayer: moeSsmAttention.alwaysPerLayer,
    expertsPerLayerTotal: moeAttentionOnly.expertsPerLayerTotal,
    expertsSsmPerLayerTotal: moeSsmAttention.expertsPerLayerTotal,
    expertsActivePerLayer: moeAttentionOnly.expertsActivePerLayer,
    expertsSsmActivePerLayer: moeSsmAttention.expertsActivePerLayer,
    denseTotal,
    moeExpertTotal,
    moeAlwaysTotal,
    moeTotal,
    totalParams,
    denseActive,
    moeActive,
    totalActive,
    denseActivePct,
    moeActivePct,
    moeInactivePerToken,
    totalMlp,
    totalAttn,
    totalAlwaysActive,
    alwaysActivePct,
    moeExpertsOnly,
    moeExpertsPct,
    invalidShapeWarnings,
    totalLayersComputed: denseLayers + moeLayers,
  };
}

function renderRow(code, title, valueStr, numericEq, lettersEq, stableRef) {
  const stableRefHtml = stableRef ? ` <span class="stable-ref">${stableRef}</span>` : '';
  const stableRefAttr = stableRef ? ` data-stable-ref="${stableRef}"` : '';
  return (
    `<div class="result-row"${stableRefAttr}>` +
      '<div class="row-line">' +
        `<div class="left"><span class="result-title">${code}: ${title}${stableRefHtml}</span></div>` +
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
  if (r.invalidShapeWarnings.length > 0) {
    const warningText = r.invalidShapeWarnings
      .map(([code, entries]) => `${code}: line${entries.length === 1 ? '' : 's'} ${entries.map(e => e.line).join(', ')}`)
      .join('; ');
    html += `<div class="info-text">Invalid shape entries were ignored in ${warningText}.</div>`;
  }
  html += '<table class="results-table"><tbody>';
  html += `<tr><td>Exact total param count</td><td>${fmt(r.totalParams)}</td></tr>`;
  html += `<tr><td>Exact active param count</td><td>${fmt(r.totalActive)}</td></tr>`;
  html += `<tr><td>Total always-active param count</td><td>${fmt(r.totalAlwaysActive)}</td></tr>`;
  html += `<tr><td>MoE layers total always-active param count</td><td>${fmt(r.moeAlwaysTotal)}</td></tr>`;
  html += `<tr><td>Always-active share of active (%)</td><td>${r.alwaysActivePct.toFixed(4)}%</td></tr>`;
  html += `<tr><td>MoE experts active param count</td><td>${fmt(r.moeExpertsOnly)}</td></tr>`;
  html += `<tr><td>MoE share of active (%)</td><td>${r.moeExpertsPct.toFixed(4)}%</td></tr>`;
  html += `<tr><td>Total MoE experts param count</td><td>${fmt(r.moeExpertTotal)}</td></tr>`;
  html += `<tr><td>Total MLP param count</td><td>${fmt(r.totalMlp)}</td></tr>`;
  html += `<tr><td>MoE inactive per token param count</td><td>${fmt(r.moeInactivePerToken)}</td></tr>`;
  html += `<tr><td>Total attention/SSM param count</td><td>${fmt(r.totalAttn)}</td></tr>`;
  html += '</tbody></table>';
  return html;
}

function renderDenseBucketRow(code, bucket, suffix) {
  const numeric = `${fmt(bucket.norms)} + ${fmt(bucket.attn)} + ${fmt(bucket.ffn)} = ${fmt(bucket.perLayer)}`;
  const refKey = bucket.label === 'Dense attention-only'
    ? 'explanation_dense_attention_only'
    : 'explanation_dense_ssm_attention';
  const detail = bucket.label === 'Dense attention-only' ? 'G + H + I' : 'J + K + L';
  return renderRow(code, `${bucket.label} per-layer params${suffix || ''}`, fmt(bucket.perLayer), numeric, detail, STABLE_LABEL_REFS[refKey]);
}

function getSharedPerLayerFormula(r) {
  if (!r.hasShared) return '0';
  if (r.sharedScope === 'per_layer') return 'P';
  return 'P ÷ (C + D)';
}

function renderMoeAlwaysRow(code, bucket, r) {
  const numeric = `${fmt(bucket.attn)} + ${fmt(bucket.normsTrans)} + ${fmt(bucket.sharedFfn)} + ${fmt(bucket.sharedPerLayer)} = ${fmt(bucket.alwaysPerLayer)}`;
  const refKey = bucket.label === 'MoE attention-only'
    ? 'explanation_moe_attention_only_always'
    : 'explanation_moe_ssm_attention_always';
  const shared = getSharedPerLayerFormula(r);
  const detail = bucket.label === 'MoE attention-only' ? `Q + R + S + ${shared}` : `U + V + W + ${shared}`;
  return renderRow(code, `${bucket.label} always-active per-layer params`, fmt(bucket.alwaysPerLayer), numeric, detail, STABLE_LABEL_REFS[refKey]);
}

function renderMoeExpertsRow(code, bucket, r) {
  const numeric = r.expertsIncludeDim
    ? `${fmt(bucket.expertsInput)} = ${fmt(bucket.expertsPerLayerTotal)}`
    : `${fmt(bucket.expertsInput)} × ${fmt(r.expertsPer)} = ${fmt(bucket.expertsPerLayerTotal)}`;
  const refKey = bucket.label === 'MoE attention-only'
    ? 'explanation_moe_attention_only_experts'
    : 'explanation_moe_ssm_attention_experts';
  const base = bucket.label === 'MoE attention-only' ? 'T' : 'X';
  const detail = r.expertsIncludeDim ? base : `M × ${base}`;
  return renderRow(code, `${bucket.label} experts per-layer params`, fmt(bucket.expertsPerLayerTotal), numeric, detail, STABLE_LABEL_REFS[refKey]);
}

function renderExplanation(r) {
  let html = '<h2>Explanation</h2>';
  const shared = getSharedPerLayerFormula(r);
  html += renderDenseBucketRow('EA', r.denseAttentionOnly);
  html += renderDenseBucketRow('EB', r.denseSsmAttention);

  const denseTotalNum = `${fmt(r.denseAttentionLayers)} × ${fmt(r.denseAttentionOnly.perLayer)} + ${fmt(r.denseSsmAttentionLayers)} × ${fmt(r.denseSsmAttention.perLayer)} = ${fmt(r.denseTotal)}`;
  html += renderRow('EC', 'Dense layers total params', fmt(r.denseTotal), denseTotalNum, 'A × EA + B × EB', STABLE_LABEL_REFS.explanation_dense_total);

  html += renderMoeAlwaysRow('ED', r.moeAttentionOnly, r);
  html += renderMoeAlwaysRow('EE', r.moeSsmAttention, r);
  html += renderMoeExpertsRow('EF', r.moeAttentionOnly, r);
  html += renderMoeExpertsRow('EG', r.moeSsmAttention, r);

  const moeTotalNum = `${fmt(r.moeAttentionLayers)} × (${fmt(r.moeAttentionOnly.alwaysPerLayer)} + ${fmt(r.moeAttentionOnly.expertsPerLayerTotal)}) + ${fmt(r.moeSsmAttentionLayers)} × (${fmt(r.moeSsmAttention.alwaysPerLayer)} + ${fmt(r.moeSsmAttention.expertsPerLayerTotal)}) = ${fmt(r.moeTotal)}`;
  html += renderRow('EH', 'MoE layers total params', fmt(r.moeTotal), moeTotalNum, 'C × (ED + EF) + D × (EE + EG)', STABLE_LABEL_REFS.explanation_moe_total);

  const moeActiveAttentionNum = r.expertsIncludeDim
    ? (r.expertsPer > 0
        ? `${fmt(r.moeAttentionOnly.expertsInput)} × (${fmt(r.activeExpertsClamped)} ÷ ${fmt(r.expertsPer)})`
        : `${fmt(r.moeAttentionOnly.expertsInput)} × 0`)
    : `${fmt(r.moeAttentionOnly.expertsInput)} × ${fmt(r.activeExpertsClamped)}`;
  const moeActiveSsmNum = r.expertsIncludeDim
    ? (r.expertsPer > 0
        ? `${fmt(r.moeSsmAttention.expertsInput)} × (${fmt(r.activeExpertsClamped)} ÷ ${fmt(r.expertsPer)})`
        : `${fmt(r.moeSsmAttention.expertsInput)} × 0`)
    : `${fmt(r.moeSsmAttention.expertsInput)} × ${fmt(r.activeExpertsClamped)}`;
  const moeActiveNum = `${fmt(r.moeAttentionLayers)} × (${fmt(r.moeAttentionOnly.alwaysPerLayer)} + ${moeActiveAttentionNum}) + ${fmt(r.moeSsmAttentionLayers)} × (${fmt(r.moeSsmAttention.alwaysPerLayer)} + ${moeActiveSsmNum}) = ${fmt(r.moeActive)}`;
  html += renderRow('EI', 'MoE layers total active params', fmt(r.moeActive), moeActiveNum, 'C × (ED + EF × (N ÷ M)) + D × (EE + EG × (N ÷ M))', STABLE_LABEL_REFS.explanation_moe_active);

  const moeAlwaysNum = `${fmt(r.moeAttentionLayers)} × ${fmt(r.moeAttentionOnly.alwaysPerLayer)} + ${fmt(r.moeSsmAttentionLayers)} × ${fmt(r.moeSsmAttention.alwaysPerLayer)} = ${fmt(r.moeAlwaysTotal)}`;
  html += renderRow('EJ', 'MoE layers total always-active params', fmt(r.moeAlwaysTotal), moeAlwaysNum, 'C × ED + D × EE', STABLE_LABEL_REFS.explanation_moe_always_total);

  const moeInactivePerTokenNum = `${fmt(r.moeAttentionOnly.inactivePerToken)} + ${fmt(r.moeSsmAttention.inactivePerToken)} = ${fmt(r.moeInactivePerToken)}`;
  html += renderRow('EK', 'MoE inactive per token param count', fmt(r.moeInactivePerToken), moeInactivePerTokenNum, 'C × EF × (1 - N ÷ M) + D × EG × (1 - N ÷ M)', STABLE_LABEL_REFS.explanation_moe_inactive_per_token);

  const denseActiveNum = `${fmt(r.embedCount)} + ${fmt(r.preFirstCount)} + ${fmt(r.denseTotal)} = ${fmt(r.denseActive)}`;
  html += renderRow('EL', 'Dense layer(s) active param count', fmt(r.denseActive), denseActiveNum, 'E + F + EC', STABLE_LABEL_REFS.explanation_dense_active);

  const totalParamsNum = `${fmt(r.denseActive)} + ${fmt(r.moeTotal)} = ${fmt(r.totalParams)}`;
  html += renderRow('EM', 'Exact total param count', fmt(r.totalParams), totalParamsNum, 'EL + EH', STABLE_LABEL_REFS.explanation_total_params);

  const totalActiveNum = `${fmt(r.denseActive)} + ${fmt(r.moeActive)} = ${fmt(r.totalActive)}`;
  html += renderRow('EN', 'Total active param count', fmt(r.totalActive), totalActiveNum, 'EL + EI', STABLE_LABEL_REFS.explanation_total_active);

  const totalAlwaysActiveNum = `${fmt(r.denseActive)} + ${fmt(r.moeAlwaysTotal)} = ${fmt(r.totalAlwaysActive)}`;
  html += renderRow('EO', 'Total always-active param count', fmt(r.totalAlwaysActive), totalAlwaysActiveNum, 'EL + EJ', STABLE_LABEL_REFS.explanation_total_always_active);

  const alwaysShareNum = r.totalActive > 0
    ? `${fmt(r.totalAlwaysActive)} ÷ ${fmt(r.totalActive)} × 100 = ${r.alwaysActivePct.toFixed(4)}%`
    : `total active = 0, so ${r.alwaysActivePct.toFixed(4)}%`;
  html += renderRow('EP', 'Always-active share of active (%)', `${r.alwaysActivePct.toFixed(4)}`, alwaysShareNum, 'EO ÷ EN × 100', STABLE_LABEL_REFS.explanation_always_share);

  const moeShareNum = r.totalActive > 0
    ? `${fmt(r.moeExpertsOnly)} ÷ ${fmt(r.totalActive)} × 100 = ${r.moeExpertsPct.toFixed(4)}%`
    : `total active = 0, so ${r.moeExpertsPct.toFixed(4)}%`;
  html += renderRow('EQ', 'MoE share of active (%)', `${r.moeExpertsPct.toFixed(4)}`, moeShareNum, 'ET ÷ EN × 100', STABLE_LABEL_REFS.explanation_moe_share_active);

  const totalMlpNum = `${fmt(r.denseAttentionLayers)} × ${fmt(r.denseAttentionOnly.ffn)} + ${fmt(r.denseSsmAttentionLayers)} × ${fmt(r.denseSsmAttention.ffn)} + ${fmt(r.moeAttentionOnly.mlpTotal)} + ${fmt(r.moeSsmAttention.mlpTotal)} = ${fmt(r.totalMlp)}`;
  html += renderRow('ER', 'Total MLP param count', fmt(r.totalMlp), totalMlpNum, `A × I + B × L + C × (S + ${shared} + EF) + D × (W + ${shared} + EG)`, STABLE_LABEL_REFS.explanation_total_mlp);

  const totalAttnNum = `${fmt(r.denseAttentionLayers)} × ${fmt(r.denseAttentionOnly.attn)} + ${fmt(r.denseSsmAttentionLayers)} × ${fmt(r.denseSsmAttention.attn)} + ${fmt(r.moeAttentionLayers)} × ${fmt(r.moeAttentionOnly.attn)} + ${fmt(r.moeSsmAttentionLayers)} × ${fmt(r.moeSsmAttention.attn)} = ${fmt(r.totalAttn)}`;
  html += renderRow('ES', 'Total attention/SSM param count', fmt(r.totalAttn), totalAttnNum, 'A × H + B × K + C × Q + D × U', STABLE_LABEL_REFS.explanation_total_attn);

  const moeExpertsActiveNum = r.expertsIncludeDim
    ? (r.expertsPer > 0
        ? `(${fmt(r.moeAttentionLayers)} × ${fmt(r.moeAttentionOnly.expertsInput)} + ${fmt(r.moeSsmAttentionLayers)} × ${fmt(r.moeSsmAttention.expertsInput)}) × (${fmt(r.activeExpertsClamped)} ÷ ${fmt(r.expertsPer)}) = ${fmt(r.moeExpertsOnly)}`
        : `0 = ${fmt(r.moeExpertsOnly)}`)
    : `${fmt(r.moeAttentionLayers)} × ${fmt(r.moeAttentionOnly.expertsInput)} × ${fmt(r.activeExpertsClamped)} + ${fmt(r.moeSsmAttentionLayers)} × ${fmt(r.moeSsmAttention.expertsInput)} × ${fmt(r.activeExpertsClamped)} = ${fmt(r.moeExpertsOnly)}`;
  html += renderRow('ET', 'MoE experts active param count', fmt(r.moeExpertsOnly), moeExpertsActiveNum, '(C × EF + D × EG) × (N ÷ M)', STABLE_LABEL_REFS.explanation_moe_experts_active);

  const moeExpertsTotalNum = `${fmt(r.moeAttentionOnly.expertTotal)} + ${fmt(r.moeSsmAttention.expertTotal)} = ${fmt(r.moeExpertTotal)}`;
  html += renderRow('EU', 'MoE experts total param count', fmt(r.moeExpertTotal), moeExpertsTotalNum, 'C × EF + D × EG', STABLE_LABEL_REFS.explanation_moe_experts_total);

  return html;
}

function getFormData() {
  return {
    total_layers: document.getElementById('total_layers')?.value || '0',
    dense_attention_layers: document.getElementById('dense_attention_layers')?.value || '0',
    dense_ssm_attention_layers: document.getElementById('dense_ssm_attention_layers')?.value || '0',
    moe_attention_layers: document.getElementById('moe_attention_layers')?.value || '0',
    moe_ssm_attention_layers: document.getElementById('moe_ssm_attention_layers')?.value || '0',
    embedding_shapes: document.getElementById('embedding_shapes')?.value || '',
    pre_first_norms: document.getElementById('pre_first_norms')?.value || '',
    dense_norms: document.getElementById('dense_norms')?.value || '',
    dense_attn: document.getElementById('dense_attn')?.value || '',
    dense_ffn: document.getElementById('dense_ffn')?.value || '',
    dense_ssm_norms: document.getElementById('dense_ssm_norms')?.value || '',
    dense_ssm_attn: document.getElementById('dense_ssm_attn')?.value || '',
    dense_ssm_ffn: document.getElementById('dense_ssm_ffn')?.value || '',
    experts_per_layer: document.getElementById('experts_per_layer')?.value || '0',
    active_experts: document.getElementById('active_experts')?.value || '0',
    has_shared_expert: document.getElementById('has_shared_expert')?.checked || false,
    shared_expert_scope: document.getElementById('shared_expert_scope')?.value || 'per_layer',
    shared_expert_tensors: document.getElementById('shared_expert_tensors')?.value || '',
    moe_attn: document.getElementById('moe_attn')?.value || '',
    moe_transitional: document.getElementById('moe_transitional')?.value || '',
    moe_shared_ffn: document.getElementById('moe_shared_ffn')?.value || '',
    moe_experts: document.getElementById('moe_experts')?.value || '',
    moe_ssm_attn: document.getElementById('moe_ssm_attn')?.value || '',
    moe_ssm_transitional: document.getElementById('moe_ssm_transitional')?.value || '',
    moe_ssm_shared_ffn: document.getElementById('moe_ssm_shared_ffn')?.value || '',
    moe_ssm_experts: document.getElementById('moe_ssm_experts')?.value || '',
    experts_include_dim: document.getElementById('experts_include_dim')?.checked || false,
  };
}

function updateComputedTotalLayers() {
  const da = parseCount(document.getElementById('dense_attention_layers')?.value);
  const ds = parseCount(document.getElementById('dense_ssm_attention_layers')?.value);
  const ma = parseCount(document.getElementById('moe_attention_layers')?.value);
  const ms = parseCount(document.getElementById('moe_ssm_attention_layers')?.value);
  const aEl = document.getElementById('total_layers');
  if (aEl) aEl.value = String(da + ds + ma + ms);
}

function calculateAndRender(opts = {}) {
  const { scroll = false } = opts;
  const resultsBox = document.getElementById('results');
  const explanationBox = document.getElementById('explanation');
  const r = computeResults(getFormData());
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

function clearSplitOnlyFields() {
  const ids = [
    'dense_ssm_attention_layers',
    'moe_ssm_attention_layers',
    'dense_ssm_norms',
    'dense_ssm_attn',
    'dense_ssm_ffn',
    'moe_ssm_attn',
    'moe_ssm_transitional',
    'moe_ssm_shared_ffn',
    'moe_ssm_experts',
  ];
  ids.forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    if (el.type === 'checkbox') {
      el.checked = false;
    } else {
      el.value = '';
    }
  });
}

function setSectionOpen(sectionId, shouldOpen) {
  const details = document.getElementById(sectionId);
  if (details) details.open = !!shouldOpen;
}

function hasTextValue(id) {
  const el = document.getElementById(id);
  return !!(el && String(el.value || '').trim());
}

function expandLayerTensorSections() {
  const denseAttentionCount = parseCount(document.getElementById('dense_attention_layers')?.value);
  const denseSsmCount = parseCount(document.getElementById('dense_ssm_attention_layers')?.value);
  const moeAttentionCount = parseCount(document.getElementById('moe_attention_layers')?.value);
  const moeSsmCount = parseCount(document.getElementById('moe_ssm_attention_layers')?.value);

  setSectionOpen(
    'dense-attention-section',
    denseAttentionCount > 0 || hasTextValue('dense_norms') || hasTextValue('dense_attn') || hasTextValue('dense_ffn'),
  );
  setSectionOpen(
    'dense-ssm-section',
    denseSsmCount > 0 || hasTextValue('dense_ssm_norms') || hasTextValue('dense_ssm_attn') || hasTextValue('dense_ssm_ffn'),
  );
  setSectionOpen(
    'moe-attention-section',
    moeAttentionCount > 0 || hasTextValue('moe_attn') || hasTextValue('moe_transitional') || hasTextValue('moe_shared_ffn') || hasTextValue('moe_experts'),
  );
  setSectionOpen(
    'moe-ssm-section',
    moeSsmCount > 0 || hasTextValue('moe_ssm_attn') || hasTextValue('moe_ssm_transitional') || hasTextValue('moe_ssm_shared_ffn') || hasTextValue('moe_ssm_experts'),
  );
}

async function prefillParamModel(model) {
  const setVal = (id, v) => {
    const el = document.getElementById(id);
    if (el) el.value = v;
  };
  const setChecked = (id, v) => {
    const el = document.getElementById(id);
    if (el) el.checked = v;
  };

  applyStableRefPresetData(createEmptyInput(), setVal, setChecked);
  clearSplitOnlyFields();
  const applied = await applyPresetModel(model, setVal, setChecked);
  if (!applied) return;
  updateSharedState();

  updateComputedTotalLayers();
  calculateAndRender({ scroll: false });
}

if (typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', () => {
    loadPresetData().catch(() => {});
    const form = document.getElementById('paramcalc-form');
    const btn = document.getElementById('calculate-btn');
    const debouncedCalc = debounce(() => calculateAndRender({ scroll: false }), 250);
    const layerCountIds = new Set([
      'dense_attention_layers',
      'dense_ssm_attention_layers',
      'moe_attention_layers',
      'moe_ssm_attention_layers',
    ]);

    form?.addEventListener('submit', (e) => {
      e.preventDefault();
      calculateAndRender({ scroll: true });
    });
    btn?.addEventListener('click', (e) => {
      e.preventDefault();
      calculateAndRender({ scroll: true });
    });

    const inputs = document.querySelectorAll('#paramcalc-form input, #paramcalc-form textarea, #paramcalc-form select');
    inputs.forEach((el) => {
      el.addEventListener('input', () => {
        if (layerCountIds.has(el.id)) updateComputedTotalLayers();
        debouncedCalc();
      });
      el.addEventListener('change', () => {
        if (layerCountIds.has(el.id)) updateComputedTotalLayers();
        debouncedCalc();
      });
    });

    const chk = document.getElementById('has_shared_expert');
    chk?.addEventListener('change', () => {
      updateSharedState();
      debouncedCalc();
    });
    updateSharedState();
    updateComputedTotalLayers();

    const presetSel = document.getElementById('param-model-select');
    presetSel?.addEventListener('change', async () => {
      if (presetSel.value === 'custom') return;
      await prefillParamModel(presetSel.value);
      expandLayerTensorSections();
    });
  });
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    parseDimToken,
    parseShapeGroup,
    sumShapes,
    collectInvalidShapeEntries,
    computeResults,
    renderSummary,
    renderExplanation,
    createEmptyInput,
    buildPresetInput,
    getPresetModels,
    STABLE_LABEL_REFS,
  };
}
