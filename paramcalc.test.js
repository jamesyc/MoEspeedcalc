const test = require('node:test');
const assert = require('node:assert/strict');

const {
  sumShapes,
  computeResults,
  renderSummary,
  renderExplanation,
} = require('./paramcalc.js');

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
    ['H', [{ line: 1, value: '[D, 4D]' }]],
  ]);

  const summary = renderSummary(r);
  assert.match(summary, /Invalid shape entries were ignored in H: line 1\./);
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
  assert.match(explanation, /C × Q = AH/);
  assert.match(explanation, /C × Q × 0 = AQ/);
  assert.match(explanation, /AK = 0, so 0.0000%/);
});

test('summary labels match AQ and AR quantities', () => {
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
