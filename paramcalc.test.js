const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');

const {
  sumShapes,
  computeResults,
  renderSummary,
  renderLetterResults,
  renderExplanation,
  buildPresetInput,
  getPresetModels,
  STABLE_LABEL_REFS,
} = require('./paramcalc.js');

const PRESET_JSON = JSON.parse(fs.readFileSync(require.resolve('./paramcalc.presets.json'), 'utf8'));
const LEGACY_BASELINES = JSON.parse(fs.readFileSync(require.resolve('./paramcalc.legacy-baselines.json'), 'utf8'));

function makeInput(overrides = {}) {
  return {
    total_layers: '0',
    dense_layers: '0',
    moe_layers: '0',
    embedding_shapes: '',
    pre_first_norms: '',
    dense_norms: '',
    dense_attn: '',
    dense_ffn: '',
    experts_per_layer: '0',
    active_experts: '0',
    has_shared_expert: false,
    shared_expert_scope: 'per_layer',
    shared_expert_tensors: '',
    moe_attn: '',
    moe_transitional: '',
    moe_shared_ffn: '',
    moe_experts: '',
    experts_include_dim: false,
    ...overrides,
  };
}

test('sumShapes supports scientific notation without truncation', () => {
  assert.equal(sumShapes('[2e3, 4]'), 8000);
  assert.equal(sumShapes('[4.096e3, 2]'), 8192);
});

test('invalid symbolic shapes are ignored and surfaced as warnings', () => {
  const r = computeResults(makeInput({
    dense_layers: '1',
    dense_ffn: '[D, 4D]\n[4096, 16384]',
  }));

  assert.equal(r.dFfn, 4096 * 16384);
  assert.deepEqual(r.invalidShapeWarnings, [
    ['I', [{ line: 1, value: '[D, 4D]' }]],
  ]);

  const summary = renderSummary(r);
  assert.match(summary, /Invalid shape entries were ignored in I: line 1\./);
});

test('zero experts per layer does not render divide-by-zero explanations', () => {
  const r = computeResults(makeInput({
    moe_layers: '2',
    experts_per_layer: '0',
    active_experts: '4',
    moe_experts: '[128, 64]',
    experts_include_dim: true,
  }));

  const explanation = renderExplanation(r);
  assert.doesNotMatch(explanation, /÷ 0/);
  assert.match(explanation, /2 × \(0 \+ 8,192 × 0\) \+ 0 × \(0 \+ 0 × 0\) = 0/);
  assert.match(explanation, /ET: MoE experts active param count/);
  assert.match(explanation, /total active = 0, so 0.0000%/);
});

test('summary labels match AO and AP quantities', () => {
  const r = computeResults(makeInput({
    moe_layers: '1',
    experts_per_layer: '4',
    active_experts: '2',
    moe_experts: '[4, 10]',
    experts_include_dim: true,
  }));

  const summary = renderSummary(r);
  assert.match(summary, /MoE experts active param count/);
  assert.match(summary, /Total MoE experts param count/);
  assert.doesNotMatch(summary, />MoE active param count</);
  assert.doesNotMatch(summary, />Total MoE param count</);
});

test('letter results render the A-X sums between summary and explanation', () => {
  const r = computeResults(makeInput({
    dense_attention_layers: '2',
    dense_ssm_attention_layers: '1',
    moe_attention_layers: '3',
    moe_ssm_attention_layers: '4',
    embedding_shapes: '[10, 20]',
    pre_first_norms: '[5]',
    dense_norms: '[7]',
    dense_attn: '[11]',
    dense_ffn: '[13]',
    dense_ssm_norms: '[17]',
    dense_ssm_attn: '[19]',
    dense_ssm_ffn: '[23]',
    experts_per_layer: '8',
    active_experts: '10',
    shared_expert_tensors: '[29]',
    moe_attn: '[31]',
    moe_transitional: '[37]',
    moe_shared_ffn: '[41]',
    moe_experts: '[43]',
    moe_ssm_attn: '[47]',
    moe_ssm_transitional: '[53]',
    moe_ssm_shared_ffn: '[59]',
    moe_ssm_experts: '[61]',
    experts_include_dim: true,
  }));

  const letterResults = renderLetterResults(r);

  assert.match(letterResults, /<h2>Results \(A-Z Sums\)<\/h2>/);
  assert.match(letterResults, /<td>A<\/td><td>Dense attention-only layers<\/td><td>2<\/td>/);
  assert.match(letterResults, /<td>E<\/td><td>Embedding\/output matrix sum<\/td><td>200<\/td>/);
  assert.match(letterResults, /<td>N<\/td><td>Active experts per MoE layer used in formulas<\/td><td>8<\/td>/);
  assert.match(letterResults, /<td>P<\/td><td>Shared expert tensors sum<\/td><td>0<\/td>/);
  assert.match(letterResults, /<td>X<\/td><td>MoE SSM\+attention experts tensors sum<\/td><td>61<\/td>/);
  assert.match(letterResults, /input active experts value was clamped/);
});

test('kimi-k2 zero-count MoE SSM bucket does not show shared-expert carryover', () => {
  const explanation = renderExplanation(computeResults(buildPresetInput('kimi-k2')));
  assert.match(explanation, /EE: MoE SSM\+attention always-active per-layer params[\s\S]*?0 \+ 0 \+ 0 \+ 0 = 0/);
});

test('per-expert presets explicitly disable tensors-include-E checkbox', () => {
  assert.equal(PRESET_JSON.models['kimi-k2'].Z44, false);
  assert.equal(PRESET_JSON.models['deepseek-v3'].Z44, false);
  assert.equal(PRESET_JSON.models['deepseek-v3-mtp'].Z44, false);
});

test('minimax has no separate mtp preset entry', () => {
  assert.equal(PRESET_JSON.models['minimax-m2.5-mtp'], undefined);
  assert.equal(PRESET_JSON.modelOrder.includes('minimax-m2.5-mtp'), false);
});

test('glm-4.7 keeps the base profile while glm-4.7-mtp includes the Hugging Face mtp block', () => {
  const baseResults = computeResults(buildPresetInput('glm-4.7'));
  assert.equal(baseResults.totalParams, 352797829024);

  const results = computeResults(buildPresetInput('glm-4.7-mtp'));
  assert.equal(results.totalParams, 358337791296);
  assert.equal(results.moeAttentionLayers, 90);
  assert.equal(results.preFirstCount, 1604341760);
});

test('summary and explanation include restored MoE aggregate rows', () => {
  const r = computeResults(buildPresetInput('kimi-k2'));
  const summary = renderSummary(r);
  const explanation = renderExplanation(r);

  assert.match(summary, /MoE layers total always-active param count/);
  assert.match(summary, /MoE inactive per token param count/);

  assert.match(explanation, /EJ: MoE layers total always-active params/);
  assert.match(explanation, /EK: MoE inactive per token param count/);
  assert.match(explanation, /EQ: MoE share of active \(%\)/);
  assert.match(explanation, /ET: MoE experts active param count/);
  assert.match(explanation, /EU: MoE experts total param count/);
});

test('form labels expose stable Z refs in order', () => {
  const html = fs.readFileSync(require.resolve('./paramcalc.html'), 'utf8');
  for (let n = 1; n <= 24; n += 1) {
    const ref = `Z${String(n).padStart(2, '0')}`;
    assert.match(html, new RegExp(`data-stable-ref="${ref}"`));
  }
  assert.match(html, /data-stable-ref="Z43"/);
  assert.match(html, /data-stable-ref="Z44"/);
  assert.match(html, /id="letter-results"/);
});

test('preset json is keyed by stable Z refs', () => {
  const allowedRefs = new Set(Object.values(STABLE_LABEL_REFS));
  assert.deepEqual(PRESET_JSON.modelOrder, getPresetModels());

  for (const model of PRESET_JSON.modelOrder) {
    const preset = PRESET_JSON.models[model];
    assert.ok(preset, `${model} missing from preset json`);
    for (const ref of Object.keys(preset)) {
      assert.match(ref, /^Z\d{2}$/);
      assert.ok(allowedRefs.has(ref), `${model} uses unknown stable ref ${ref}`);
    }
  }
});

test('preset models preserve explanation stable refs and match locked baseline outputs', () => {
  const expectedExplanationRefs = Object.entries(STABLE_LABEL_REFS)
    .filter(([key]) => key.startsWith('explanation_'))
    .map(([, ref]) => ref);

  for (const model of getPresetModels()) {
    const input = buildPresetInput(model);
    const results = computeResults(input);
    const explanation = renderExplanation(results);

    for (const ref of expectedExplanationRefs) {
      assert.match(explanation, new RegExp(`data-stable-ref="${ref}"`), `${model} missing ${ref}`);
    }

    assert.deepEqual({
      totalParams: results.totalParams,
      totalActive: results.totalActive,
      totalAlwaysActive: results.totalAlwaysActive,
      moeExpertTotal: results.moeExpertTotal,
      moeExpertsOnly: results.moeExpertsOnly,
      totalMlp: results.totalMlp,
      totalAttn: results.totalAttn,
      totalLayersComputed: results.totalLayersComputed,
    }, LEGACY_BASELINES.models[model], `${model} no longer matches locked baseline output`);
  }
});
