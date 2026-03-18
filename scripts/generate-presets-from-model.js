#!/usr/bin/env node

const fs = require('node:fs');
const path = require('node:path');

const {
  STABLE_LABEL_REFS,
  createEmptyInput,
  computeResults,
} = require('../paramcalc.js');

const FIELD_IDS_BY_STABLE_REF = Object.fromEntries(
  Object.entries(STABLE_LABEL_REFS)
    .filter(([fieldId]) => !fieldId.startsWith('explanation_'))
    .map(([fieldId, stableRef]) => [stableRef, fieldId]),
);

const EXISTING_PRESETS = JSON.parse(
  fs.readFileSync(path.join(__dirname, '..', 'paramcalc.presets.json'), 'utf8'),
).models;

function requireTensor(model, name) {
  const tensor = model.tensors[name];
  if (!tensor) {
    throw new Error(`Missing tensor ${name}`);
  }
  return tensor;
}

function maybeTensor(model, name) {
  return model.tensors[name] || null;
}

function product(shape) {
  return shape.reduce((total, dim) => total * dim, 1);
}

function shapeLine(shape) {
  return `[${shape.join(', ')}]`;
}

function scalarLine(total) {
  return shapeLine([total]);
}

function tensorLine(model, name) {
  return shapeLine(requireTensor(model, name).shape);
}

function tensorCount(model, name) {
  return product(requireTensor(model, name).shape);
}

function linesFromNames(model, names) {
  return names.map(name => tensorLine(model, name)).join('\n');
}

function linesFromExistingNames(model, names) {
  return names
    .filter(name => maybeTensor(model, name))
    .map(name => tensorLine(model, name))
    .join('\n');
}

function nonEmptyJoin(lines) {
  return lines.filter(Boolean).join('\n');
}

function sumTensorCounts(model, names) {
  return names.reduce((total, name) => total + tensorCount(model, name), 0);
}

function listTensorNames(model) {
  return Object.keys(model.tensors || {});
}

function tensorNamesWithPrefix(model, prefix) {
  return listTensorNames(model)
    .filter(name => name.startsWith(prefix))
    .sort();
}

function lineBlockFromPrefix(model, prefix) {
  return linesFromNames(model, tensorNamesWithPrefix(model, prefix));
}

function scalarLineForPrefix(model, prefix) {
  return scalarLine(
    tensorNamesWithPrefix(model, prefix).reduce((total, name) => total + tensorCount(model, name), 0),
  );
}

function hasPrefix(model, prefix) {
  return tensorNamesWithPrefix(model, prefix).length > 0;
}

function getGroupIndices(model, groupName) {
  return model.groups?.[groupName]?.layer_indices || [];
}

function getLayerIndicesForPrefix(model, prefix) {
  const re = new RegExp(`^${prefix.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\.(\\d+)\\.`);
  return [...new Set(
    listTensorNames(model)
      .map(name => name.match(re))
      .filter(Boolean)
      .map(match => Number(match[1])),
  )].sort((a, b) => a - b);
}

function getExpertCountFromRouter(model, name) {
  return requireTensor(model, name).shape[0];
}

function getExpertCountFromIndexedExperts(model, prefix) {
  const re = new RegExp(`^${prefix.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(\\d+)\\.`);
  const ids = listTensorNames(model)
    .map(name => name.match(re))
    .filter(Boolean)
    .map(match => Number(match[1]));
  if (ids.length === 0) {
    throw new Error(`Could not infer expert count from ${prefix}*`);
  }
  return Math.max(...ids) + 1;
}

function expertLine(model, tensorName, experts) {
  return shapeLine([experts, ...requireTensor(model, tensorName).shape]);
}

function buildPresetInput(entry) {
  const input = createEmptyInput();
  for (const [stableRef, value] of Object.entries(entry)) {
    const fieldId = FIELD_IDS_BY_STABLE_REF[stableRef];
    if (!fieldId) continue;
    input[fieldId] = value;
  }
  return input;
}

function inferActiveExperts(model, presetId) {
  if (model.architecture?.num_experts_per_tok != null) {
    return model.architecture.num_experts_per_tok;
  }
  const existing = EXISTING_PRESETS[presetId];
  if (existing?.[STABLE_LABEL_REFS.active_experts] != null) {
    return Number(existing[STABLE_LABEL_REFS.active_experts]);
  }
  throw new Error(`Could not infer active experts for ${presetId}`);
}

function buildEntriesFromPresetExports(model, buildEntry) {
  const presetExports = model.preset_exports || {};
  const hasActualMtpTensors = hasPrefix(model, 'mtp.');
  const presetIds = Object.keys(presetExports).filter(presetId => {
    const presetExport = presetExports[presetId] || {};
    const includeMtp = Boolean(presetExport.include_groups?.includes('mtp_layers'));
    if (!includeMtp) return true;
    if (hasActualMtpTensors) return true;
    return Number(presetExport.mtp_only_parameter_delta || 0) !== 0;
  });
  const models = {};
  const expectedTotals = {};

  for (const presetId of presetIds) {
    const includeMtp = Boolean(presetExports[presetId]?.include_groups?.includes('mtp_layers'));
    const { entry, expectedTotal } = buildEntry(presetId, includeMtp);
    models[presetId] = entry;
    if (expectedTotal != null) {
      expectedTotals[presetId] = expectedTotal;
    }
  }

  return { models, expectedTotals };
}

function buildDeepseekStyleEntry(model, presetId, includeMtp) {
  const denseLayer = getGroupIndices(model, 'dense_layers')[0];
  const moeLayer = getGroupIndices(model, 'moe_layers')[0];
  const mtpLayer = getGroupIndices(model, 'mtp_layers')[0];
  const denseCount = getGroupIndices(model, 'dense_layers').length;
  const moeCount = getGroupIndices(model, 'moe_layers').length + (includeMtp && mtpLayer != null ? 1 : 0);
  const experts = model.architecture.n_routed_experts;
  const isKimi = model.architecture.model_type === 'kimi_k2';

  const z05 = [
    tensorLine(model, 'model.embed_tokens.weight'),
    tensorLine(model, 'lm_head.weight'),
  ];
  const z06 = [tensorLine(model, 'model.norm.weight')];

  if (includeMtp && mtpLayer != null) {
    if (!isKimi) {
      z05.push(
        tensorLine(model, `model.layers.${mtpLayer}.embed_tokens.weight`),
        tensorLine(model, `model.layers.${mtpLayer}.shared_head.head.weight`),
      );
      z06.push(
        tensorLine(model, `model.layers.${mtpLayer}.enorm.weight`),
        tensorLine(model, `model.layers.${mtpLayer}.hnorm.weight`),
        tensorLine(model, `model.layers.${mtpLayer}.eh_proj.weight`),
        tensorLine(model, `model.layers.${mtpLayer}.shared_head.norm.weight`),
      );
    }
  }

  const rotaryName = `model.layers.${denseLayer}.self_attn.rotary_emb.inv_freq`;
  const moeRotaryName = `model.layers.${moeLayer}.self_attn.rotary_emb.inv_freq`;
  const denseAttnNames = [
    `model.layers.${denseLayer}.self_attn.q_a_proj.weight`,
    `model.layers.${denseLayer}.self_attn.q_a_proj.weight_scale_inv`,
    `model.layers.${denseLayer}.self_attn.q_b_proj.weight`,
    `model.layers.${denseLayer}.self_attn.q_b_proj.weight_scale_inv`,
    `model.layers.${denseLayer}.self_attn.kv_a_proj_with_mqa.weight`,
    `model.layers.${denseLayer}.self_attn.kv_a_proj_with_mqa.weight_scale_inv`,
    `model.layers.${denseLayer}.self_attn.kv_b_proj.weight`,
    `model.layers.${denseLayer}.self_attn.kv_b_proj.weight_scale_inv`,
    `model.layers.${denseLayer}.self_attn.o_proj.weight`,
    `model.layers.${denseLayer}.self_attn.o_proj.weight_scale_inv`,
    `model.layers.${denseLayer}.self_attn.q_a_layernorm.weight`,
    `model.layers.${denseLayer}.self_attn.kv_a_layernorm.weight`,
  ];
  const moeAttnNames = [
    `model.layers.${moeLayer}.self_attn.q_a_proj.weight`,
    `model.layers.${moeLayer}.self_attn.q_a_proj.weight_scale_inv`,
    `model.layers.${moeLayer}.self_attn.q_b_proj.weight`,
    `model.layers.${moeLayer}.self_attn.q_b_proj.weight_scale_inv`,
    `model.layers.${moeLayer}.self_attn.kv_a_proj_with_mqa.weight`,
    `model.layers.${moeLayer}.self_attn.kv_a_proj_with_mqa.weight_scale_inv`,
    `model.layers.${moeLayer}.self_attn.kv_b_proj.weight`,
    `model.layers.${moeLayer}.self_attn.kv_b_proj.weight_scale_inv`,
    `model.layers.${moeLayer}.self_attn.o_proj.weight`,
    `model.layers.${moeLayer}.self_attn.o_proj.weight_scale_inv`,
    `model.layers.${moeLayer}.self_attn.q_a_layernorm.weight`,
    `model.layers.${moeLayer}.self_attn.kv_a_layernorm.weight`,
  ];
  if (maybeTensor(model, rotaryName)) {
    denseAttnNames.push(rotaryName);
  }
  if (maybeTensor(model, moeRotaryName)) {
    moeAttnNames.push(moeRotaryName);
  }

  return {
    entry: {
      [STABLE_LABEL_REFS.dense_attention_layers]: String(denseCount),
      [STABLE_LABEL_REFS.moe_attention_layers]: String(moeCount),
      [STABLE_LABEL_REFS.embedding_shapes]: z05.join('\n'),
      [STABLE_LABEL_REFS.pre_first_norms]: z06.join('\n'),
      [STABLE_LABEL_REFS.dense_norms]: linesFromNames(model, [
        `model.layers.${denseLayer}.input_layernorm.weight`,
        `model.layers.${denseLayer}.post_attention_layernorm.weight`,
      ]),
      [STABLE_LABEL_REFS.dense_attn]: linesFromNames(model, denseAttnNames),
      [STABLE_LABEL_REFS.dense_ffn]: linesFromNames(model, [
        `model.layers.${denseLayer}.mlp.gate_proj.weight`,
        `model.layers.${denseLayer}.mlp.gate_proj.weight_scale_inv`,
        `model.layers.${denseLayer}.mlp.up_proj.weight`,
        `model.layers.${denseLayer}.mlp.up_proj.weight_scale_inv`,
        `model.layers.${denseLayer}.mlp.down_proj.weight`,
        `model.layers.${denseLayer}.mlp.down_proj.weight_scale_inv`,
      ]),
      [STABLE_LABEL_REFS.experts_per_layer]: String(experts),
      [STABLE_LABEL_REFS.active_experts]: String(model.architecture.num_experts_per_tok),
      [STABLE_LABEL_REFS.shared_expert_scope]: 'per_layer',
      [STABLE_LABEL_REFS.shared_expert_tensors]: linesFromNames(model, [
        `model.layers.${moeLayer}.mlp.shared_experts.gate_proj.weight`,
        `model.layers.${moeLayer}.mlp.shared_experts.gate_proj.weight_scale_inv`,
        `model.layers.${moeLayer}.mlp.shared_experts.up_proj.weight`,
        `model.layers.${moeLayer}.mlp.shared_experts.up_proj.weight_scale_inv`,
        `model.layers.${moeLayer}.mlp.shared_experts.down_proj.weight`,
        `model.layers.${moeLayer}.mlp.shared_experts.down_proj.weight_scale_inv`,
      ]),
      [STABLE_LABEL_REFS.moe_attn]: linesFromNames(model, moeAttnNames),
      [STABLE_LABEL_REFS.moe_transitional]: linesFromNames(model, [
        `model.layers.${moeLayer}.input_layernorm.weight`,
        `model.layers.${moeLayer}.post_attention_layernorm.weight`,
      ]),
      [STABLE_LABEL_REFS.moe_shared_ffn]: linesFromNames(model, [
        `model.layers.${moeLayer}.mlp.gate.weight`,
        `model.layers.${moeLayer}.mlp.gate.e_score_correction_bias`,
      ]),
      [STABLE_LABEL_REFS.moe_experts]: linesFromNames(model, [
        `model.layers.${moeLayer}.mlp.experts.0.gate_proj.weight`,
        `model.layers.${moeLayer}.mlp.experts.0.gate_proj.weight_scale_inv`,
        `model.layers.${moeLayer}.mlp.experts.0.up_proj.weight`,
        `model.layers.${moeLayer}.mlp.experts.0.up_proj.weight_scale_inv`,
        `model.layers.${moeLayer}.mlp.experts.0.down_proj.weight`,
        `model.layers.${moeLayer}.mlp.experts.0.down_proj.weight_scale_inv`,
      ]),
      [STABLE_LABEL_REFS.has_shared_expert]: true,
      [STABLE_LABEL_REFS.experts_include_dim]: false,
    },
  };
}

function buildGlm4MoeEntry(model, presetId, includeMtp) {
  const denseLayer = getGroupIndices(model, 'dense_layers')[0];
  const moeLayer = getGroupIndices(model, 'moe_layers')[0];
  const mtpLayer = getGroupIndices(model, 'mtp_layers')[0];
  const denseCount = getGroupIndices(model, 'dense_layers').length;
  const moeCount = getGroupIndices(model, 'moe_layers').length + (includeMtp && mtpLayer != null ? 1 : 0);
  const experts = model.architecture.n_routed_experts;

  const z06 = [tensorLine(model, 'model.norm.weight')];
  if (includeMtp && mtpLayer != null) {
    z06.push(
      tensorLine(model, `model.layers.${mtpLayer}.embed_tokens.weight`),
      tensorLine(model, `model.layers.${mtpLayer}.eh_proj.weight`),
      tensorLine(model, `model.layers.${mtpLayer}.enorm.weight`),
      tensorLine(model, `model.layers.${mtpLayer}.hnorm.weight`),
      tensorLine(model, `model.layers.${mtpLayer}.shared_head.head.weight`),
      tensorLine(model, `model.layers.${mtpLayer}.shared_head.norm.weight`),
    );
  }

  return {
    entry: {
      [STABLE_LABEL_REFS.dense_attention_layers]: String(denseCount),
      [STABLE_LABEL_REFS.moe_attention_layers]: String(moeCount),
      [STABLE_LABEL_REFS.embedding_shapes]: linesFromNames(model, [
        'model.embed_tokens.weight',
        'lm_head.weight',
      ]),
      [STABLE_LABEL_REFS.pre_first_norms]: z06.join('\n'),
      [STABLE_LABEL_REFS.dense_norms]: linesFromNames(model, [
        `model.layers.${denseLayer}.input_layernorm.weight`,
        `model.layers.${denseLayer}.post_attention_layernorm.weight`,
        `model.layers.${denseLayer}.self_attn.q_norm.weight`,
        `model.layers.${denseLayer}.self_attn.k_norm.weight`,
      ]),
      [STABLE_LABEL_REFS.dense_attn]: linesFromNames(model, [
        `model.layers.${denseLayer}.self_attn.q_proj.weight`,
        `model.layers.${denseLayer}.self_attn.q_proj.bias`,
        `model.layers.${denseLayer}.self_attn.k_proj.weight`,
        `model.layers.${denseLayer}.self_attn.k_proj.bias`,
        `model.layers.${denseLayer}.self_attn.v_proj.weight`,
        `model.layers.${denseLayer}.self_attn.v_proj.bias`,
        `model.layers.${denseLayer}.self_attn.o_proj.weight`,
      ]),
      [STABLE_LABEL_REFS.dense_ffn]: linesFromNames(model, [
        `model.layers.${denseLayer}.mlp.gate_proj.weight`,
        `model.layers.${denseLayer}.mlp.up_proj.weight`,
        `model.layers.${denseLayer}.mlp.down_proj.weight`,
      ]),
      [STABLE_LABEL_REFS.experts_per_layer]: String(experts),
      [STABLE_LABEL_REFS.active_experts]: String(model.architecture.num_experts_per_tok),
      [STABLE_LABEL_REFS.shared_expert_scope]: 'per_layer',
      [STABLE_LABEL_REFS.shared_expert_tensors]: linesFromNames(model, [
        `model.layers.${moeLayer}.mlp.shared_experts.gate_proj.weight`,
        `model.layers.${moeLayer}.mlp.shared_experts.up_proj.weight`,
        `model.layers.${moeLayer}.mlp.shared_experts.down_proj.weight`,
      ]),
      [STABLE_LABEL_REFS.moe_attn]: linesFromNames(model, [
        `model.layers.${moeLayer}.self_attn.q_proj.weight`,
        `model.layers.${moeLayer}.self_attn.q_proj.bias`,
        `model.layers.${moeLayer}.self_attn.k_proj.weight`,
        `model.layers.${moeLayer}.self_attn.k_proj.bias`,
        `model.layers.${moeLayer}.self_attn.v_proj.weight`,
        `model.layers.${moeLayer}.self_attn.v_proj.bias`,
        `model.layers.${moeLayer}.self_attn.o_proj.weight`,
      ]),
      [STABLE_LABEL_REFS.moe_transitional]: linesFromNames(model, [
        `model.layers.${moeLayer}.input_layernorm.weight`,
        `model.layers.${moeLayer}.post_attention_layernorm.weight`,
        `model.layers.${moeLayer}.self_attn.q_norm.weight`,
        `model.layers.${moeLayer}.self_attn.k_norm.weight`,
      ]),
      [STABLE_LABEL_REFS.moe_shared_ffn]: linesFromNames(model, [
        `model.layers.${moeLayer}.mlp.gate.weight`,
        `model.layers.${moeLayer}.mlp.gate.e_score_correction_bias`,
      ]),
      [STABLE_LABEL_REFS.moe_experts]: [
        expertLine(model, `model.layers.${moeLayer}.mlp.experts.0.gate_proj.weight`, experts),
        expertLine(model, `model.layers.${moeLayer}.mlp.experts.0.up_proj.weight`, experts),
        expertLine(model, `model.layers.${moeLayer}.mlp.experts.0.down_proj.weight`, experts),
      ].join('\n'),
      [STABLE_LABEL_REFS.has_shared_expert]: true,
      [STABLE_LABEL_REFS.experts_include_dim]: true,
    },
  };
}

function buildGlm4MoeLiteEntry(model, presetId, includeMtp) {
  const denseLayer = getGroupIndices(model, 'dense_layers')[0];
  const moeLayer = getGroupIndices(model, 'moe_layers')[0];
  const mtpLayer = getGroupIndices(model, 'mtp_layers')[0];
  const denseCount = getGroupIndices(model, 'dense_layers').length;
  const moeCount = getGroupIndices(model, 'moe_layers').length + (includeMtp && mtpLayer != null ? 1 : 0);
  const experts = model.architecture.n_routed_experts;

  const z06 = [tensorLine(model, 'model.norm.weight')];
  if (includeMtp && mtpLayer != null) {
    z06.push(
      tensorLine(model, `model.layers.${mtpLayer}.embed_tokens.weight`),
      tensorLine(model, `model.layers.${mtpLayer}.eh_proj.weight`),
      tensorLine(model, `model.layers.${mtpLayer}.enorm.weight`),
      tensorLine(model, `model.layers.${mtpLayer}.hnorm.weight`),
      tensorLine(model, `model.layers.${mtpLayer}.shared_head.head.weight`),
      tensorLine(model, `model.layers.${mtpLayer}.shared_head.norm.weight`),
    );
  }

  return {
    entry: {
      [STABLE_LABEL_REFS.dense_attention_layers]: String(denseCount),
      [STABLE_LABEL_REFS.moe_attention_layers]: String(moeCount),
      [STABLE_LABEL_REFS.embedding_shapes]: linesFromNames(model, [
        'model.embed_tokens.weight',
        'lm_head.weight',
      ]),
      [STABLE_LABEL_REFS.pre_first_norms]: z06.join('\n'),
      [STABLE_LABEL_REFS.dense_norms]: linesFromNames(model, [
        `model.layers.${denseLayer}.input_layernorm.weight`,
        `model.layers.${denseLayer}.post_attention_layernorm.weight`,
        `model.layers.${denseLayer}.self_attn.q_a_layernorm.weight`,
        `model.layers.${denseLayer}.self_attn.kv_a_layernorm.weight`,
      ]),
      [STABLE_LABEL_REFS.dense_attn]: linesFromNames(model, [
        `model.layers.${denseLayer}.self_attn.q_a_proj.weight`,
        `model.layers.${denseLayer}.self_attn.q_b_proj.weight`,
        `model.layers.${denseLayer}.self_attn.kv_a_proj_with_mqa.weight`,
        `model.layers.${denseLayer}.self_attn.kv_b_proj.weight`,
        `model.layers.${denseLayer}.self_attn.o_proj.weight`,
      ]),
      [STABLE_LABEL_REFS.dense_ffn]: linesFromNames(model, [
        `model.layers.${denseLayer}.mlp.gate_proj.weight`,
        `model.layers.${denseLayer}.mlp.up_proj.weight`,
        `model.layers.${denseLayer}.mlp.down_proj.weight`,
      ]),
      [STABLE_LABEL_REFS.experts_per_layer]: String(experts),
      [STABLE_LABEL_REFS.active_experts]: String(model.architecture.num_experts_per_tok),
      [STABLE_LABEL_REFS.shared_expert_scope]: 'per_layer',
      [STABLE_LABEL_REFS.shared_expert_tensors]: linesFromNames(model, [
        `model.layers.${moeLayer}.mlp.shared_experts.gate_proj.weight`,
        `model.layers.${moeLayer}.mlp.shared_experts.up_proj.weight`,
        `model.layers.${moeLayer}.mlp.shared_experts.down_proj.weight`,
      ]),
      [STABLE_LABEL_REFS.moe_attn]: linesFromNames(model, [
        `model.layers.${moeLayer}.self_attn.q_a_proj.weight`,
        `model.layers.${moeLayer}.self_attn.q_b_proj.weight`,
        `model.layers.${moeLayer}.self_attn.kv_a_proj_with_mqa.weight`,
        `model.layers.${moeLayer}.self_attn.kv_b_proj.weight`,
        `model.layers.${moeLayer}.self_attn.o_proj.weight`,
      ]),
      [STABLE_LABEL_REFS.moe_transitional]: linesFromNames(model, [
        `model.layers.${moeLayer}.input_layernorm.weight`,
        `model.layers.${moeLayer}.post_attention_layernorm.weight`,
        `model.layers.${moeLayer}.self_attn.q_a_layernorm.weight`,
        `model.layers.${moeLayer}.self_attn.kv_a_layernorm.weight`,
      ]),
      [STABLE_LABEL_REFS.moe_shared_ffn]: linesFromNames(model, [
        `model.layers.${moeLayer}.mlp.gate.weight`,
        `model.layers.${moeLayer}.mlp.gate.e_score_correction_bias`,
      ]),
      [STABLE_LABEL_REFS.moe_experts]: [
        expertLine(model, `model.layers.${moeLayer}.mlp.experts.0.gate_proj.weight`, experts),
        expertLine(model, `model.layers.${moeLayer}.mlp.experts.0.up_proj.weight`, experts),
        expertLine(model, `model.layers.${moeLayer}.mlp.experts.0.down_proj.weight`, experts),
      ].join('\n'),
      [STABLE_LABEL_REFS.has_shared_expert]: true,
      [STABLE_LABEL_REFS.experts_include_dim]: true,
    },
  };
}

function buildGlmMoeDsaEntry(model, presetId, includeMtp) {
  const denseLayer = getGroupIndices(model, 'dense_layers')[0];
  const moeLayer = getGroupIndices(model, 'moe_layers')[0];
  const mtpLayer = getGroupIndices(model, 'mtp_layers')[0];
  const moeCount = getGroupIndices(model, 'moe_layers').length + (includeMtp && mtpLayer != null ? 1 : 0);

  const z06 = [tensorLine(model, 'model.norm.weight')];
  if (includeMtp && mtpLayer != null) {
    z06.push(
      tensorLine(model, `model.layers.${mtpLayer}.eh_proj.weight`),
      tensorLine(model, `model.layers.${mtpLayer}.enorm.weight`),
      tensorLine(model, `model.layers.${mtpLayer}.hnorm.weight`),
      tensorLine(model, `model.layers.${mtpLayer}.shared_head.norm.weight`),
    );
  }

  return {
    entry: {
      [STABLE_LABEL_REFS.dense_attention_layers]: String(getGroupIndices(model, 'dense_layers').length),
      [STABLE_LABEL_REFS.moe_attention_layers]: String(moeCount),
      [STABLE_LABEL_REFS.embedding_shapes]: linesFromNames(model, [
        'model.embed_tokens.weight',
        'lm_head.weight',
      ]),
      [STABLE_LABEL_REFS.pre_first_norms]: z06.join('\n'),
      [STABLE_LABEL_REFS.dense_norms]: linesFromNames(model, [
        `model.layers.${denseLayer}.input_layernorm.weight`,
        `model.layers.${denseLayer}.post_attention_layernorm.weight`,
        `model.layers.${denseLayer}.self_attn.q_a_layernorm.weight`,
        `model.layers.${denseLayer}.self_attn.kv_a_layernorm.weight`,
        `model.layers.${denseLayer}.self_attn.indexer.k_norm.weight`,
        `model.layers.${denseLayer}.self_attn.indexer.k_norm.bias`,
      ]),
      [STABLE_LABEL_REFS.dense_attn]: linesFromNames(model, [
        `model.layers.${denseLayer}.self_attn.q_a_proj.weight`,
        `model.layers.${denseLayer}.self_attn.q_b_proj.weight`,
        `model.layers.${denseLayer}.self_attn.kv_a_proj_with_mqa.weight`,
        `model.layers.${denseLayer}.self_attn.kv_b_proj.weight`,
        `model.layers.${denseLayer}.self_attn.o_proj.weight`,
        `model.layers.${denseLayer}.self_attn.indexer.weights_proj.weight`,
        `model.layers.${denseLayer}.self_attn.indexer.wk.weight`,
        `model.layers.${denseLayer}.self_attn.indexer.wq_b.weight`,
      ]),
      [STABLE_LABEL_REFS.dense_ffn]: linesFromNames(model, [
        `model.layers.${denseLayer}.mlp.gate_proj.weight`,
        `model.layers.${denseLayer}.mlp.up_proj.weight`,
        `model.layers.${denseLayer}.mlp.down_proj.weight`,
      ]),
      [STABLE_LABEL_REFS.experts_per_layer]: String(model.architecture.n_routed_experts),
      [STABLE_LABEL_REFS.active_experts]: String(model.architecture.num_experts_per_tok),
      [STABLE_LABEL_REFS.shared_expert_scope]: 'per_layer',
      [STABLE_LABEL_REFS.shared_expert_tensors]: linesFromNames(model, [
        `model.layers.${moeLayer}.mlp.shared_experts.gate_proj.weight`,
        `model.layers.${moeLayer}.mlp.shared_experts.up_proj.weight`,
        `model.layers.${moeLayer}.mlp.shared_experts.down_proj.weight`,
      ]),
      [STABLE_LABEL_REFS.moe_attn]: linesFromNames(model, [
        `model.layers.${moeLayer}.self_attn.q_a_proj.weight`,
        `model.layers.${moeLayer}.self_attn.q_b_proj.weight`,
        `model.layers.${moeLayer}.self_attn.kv_a_proj_with_mqa.weight`,
        `model.layers.${moeLayer}.self_attn.kv_b_proj.weight`,
        `model.layers.${moeLayer}.self_attn.o_proj.weight`,
        `model.layers.${moeLayer}.self_attn.indexer.weights_proj.weight`,
        `model.layers.${moeLayer}.self_attn.indexer.wk.weight`,
        `model.layers.${moeLayer}.self_attn.indexer.wq_b.weight`,
      ]),
      [STABLE_LABEL_REFS.moe_transitional]: linesFromNames(model, [
        `model.layers.${moeLayer}.input_layernorm.weight`,
        `model.layers.${moeLayer}.post_attention_layernorm.weight`,
        `model.layers.${moeLayer}.self_attn.q_a_layernorm.weight`,
        `model.layers.${moeLayer}.self_attn.kv_a_layernorm.weight`,
        `model.layers.${moeLayer}.self_attn.indexer.k_norm.weight`,
        `model.layers.${moeLayer}.self_attn.indexer.k_norm.bias`,
      ]),
      [STABLE_LABEL_REFS.moe_shared_ffn]: linesFromNames(model, [
        `model.layers.${moeLayer}.mlp.gate.weight`,
        `model.layers.${moeLayer}.mlp.gate.e_score_correction_bias`,
      ]),
      [STABLE_LABEL_REFS.moe_experts]: [
        expertLine(model, `model.layers.${moeLayer}.mlp.experts.0.down_proj.weight`, model.architecture.n_routed_experts),
        expertLine(model, `model.layers.${moeLayer}.mlp.experts.0.gate_proj.weight`, model.architecture.n_routed_experts),
        expertLine(model, `model.layers.${moeLayer}.mlp.experts.0.up_proj.weight`, model.architecture.n_routed_experts),
      ].join('\n'),
      [STABLE_LABEL_REFS.has_shared_expert]: true,
      [STABLE_LABEL_REFS.experts_include_dim]: true,
    },
  };
}

function buildQwen3MoeEntry(model, presetId) {
  const layerIndices = getLayerIndicesForPrefix(model, 'model.layers');
  const layer = layerIndices[0];
  const experts = getExpertCountFromRouter(model, `model.layers.${layer}.mlp.gate.weight`);

  return {
    entry: {
      [STABLE_LABEL_REFS.moe_attention_layers]: String(layerIndices.length),
      [STABLE_LABEL_REFS.embedding_shapes]: linesFromNames(model, [
        'model.embed_tokens.weight',
        'lm_head.weight',
      ]),
      [STABLE_LABEL_REFS.pre_first_norms]: tensorLine(model, 'model.norm.weight'),
      [STABLE_LABEL_REFS.experts_per_layer]: String(experts),
      [STABLE_LABEL_REFS.active_experts]: String(model.architecture.num_experts_per_tok),
      [STABLE_LABEL_REFS.shared_expert_scope]: 'per_layer',
      [STABLE_LABEL_REFS.moe_attn]: linesFromNames(model, [
        `model.layers.${layer}.self_attn.q_proj.weight`,
        `model.layers.${layer}.self_attn.k_proj.weight`,
        `model.layers.${layer}.self_attn.v_proj.weight`,
        `model.layers.${layer}.self_attn.o_proj.weight`,
      ]),
      [STABLE_LABEL_REFS.moe_transitional]: linesFromNames(model, [
        `model.layers.${layer}.input_layernorm.weight`,
        `model.layers.${layer}.post_attention_layernorm.weight`,
        `model.layers.${layer}.self_attn.q_norm.weight`,
        `model.layers.${layer}.self_attn.k_norm.weight`,
      ]),
      [STABLE_LABEL_REFS.moe_shared_ffn]: tensorLine(model, `model.layers.${layer}.mlp.gate.weight`),
      [STABLE_LABEL_REFS.moe_experts]: [
        expertLine(model, `model.layers.${layer}.mlp.experts.0.gate_proj.weight`, experts),
        expertLine(model, `model.layers.${layer}.mlp.experts.0.up_proj.weight`, experts),
        expertLine(model, `model.layers.${layer}.mlp.experts.0.down_proj.weight`, experts),
      ].join('\n'),
      [STABLE_LABEL_REFS.experts_include_dim]: true,
    },
  };
}

function buildQwen35DenseEntry(model, presetId, includeMtp) {
  const layerIndices = getLayerIndicesForPrefix(model, 'model.language_model.layers');
  const selfLayer = layerIndices.find(index => hasPrefix(model, `model.language_model.layers.${index}.self_attn`));
  const linearLayer = layerIndices.find(index => hasPrefix(model, `model.language_model.layers.${index}.linear_attn`));
  const selfCount = layerIndices.filter(index => hasPrefix(model, `model.language_model.layers.${index}.self_attn`)).length;
  const linearCount = layerIndices.length - selfCount;

  const z06 = [
    tensorLine(model, 'model.language_model.norm.weight'),
    scalarLineForPrefix(model, 'model.visual.'),
  ];
  if (includeMtp && hasPrefix(model, 'mtp.')) {
    z06.push(scalarLineForPrefix(model, 'mtp.'));
  }

  return {
    entry: {
      [STABLE_LABEL_REFS.dense_attention_layers]: String(selfCount),
      [STABLE_LABEL_REFS.dense_ssm_attention_layers]: String(linearCount),
      [STABLE_LABEL_REFS.embedding_shapes]: linesFromNames(model, [
        'model.language_model.embed_tokens.weight',
        'lm_head.weight',
      ]),
      [STABLE_LABEL_REFS.pre_first_norms]: z06.join('\n'),
      [STABLE_LABEL_REFS.dense_norms]: linesFromNames(model, [
        `model.language_model.layers.${selfLayer}.input_layernorm.weight`,
        `model.language_model.layers.${selfLayer}.post_attention_layernorm.weight`,
        `model.language_model.layers.${selfLayer}.self_attn.q_norm.weight`,
        `model.language_model.layers.${selfLayer}.self_attn.k_norm.weight`,
      ]),
      [STABLE_LABEL_REFS.dense_attn]: linesFromNames(model, [
        `model.language_model.layers.${selfLayer}.self_attn.q_proj.weight`,
        `model.language_model.layers.${selfLayer}.self_attn.k_proj.weight`,
        `model.language_model.layers.${selfLayer}.self_attn.v_proj.weight`,
        `model.language_model.layers.${selfLayer}.self_attn.o_proj.weight`,
      ]),
      [STABLE_LABEL_REFS.dense_ffn]: linesFromNames(model, [
        `model.language_model.layers.${selfLayer}.mlp.gate_proj.weight`,
        `model.language_model.layers.${selfLayer}.mlp.up_proj.weight`,
        `model.language_model.layers.${selfLayer}.mlp.down_proj.weight`,
      ]),
      [STABLE_LABEL_REFS.dense_ssm_norms]: linesFromNames(model, [
        `model.language_model.layers.${linearLayer}.input_layernorm.weight`,
        `model.language_model.layers.${linearLayer}.post_attention_layernorm.weight`,
        `model.language_model.layers.${linearLayer}.linear_attn.norm.weight`,
        `model.language_model.layers.${linearLayer}.linear_attn.A_log`,
        `model.language_model.layers.${linearLayer}.linear_attn.dt_bias`,
      ]),
      [STABLE_LABEL_REFS.dense_ssm_attn]: linesFromNames(model, [
        `model.language_model.layers.${linearLayer}.linear_attn.in_proj_qkv.weight`,
        `model.language_model.layers.${linearLayer}.linear_attn.in_proj_z.weight`,
        `model.language_model.layers.${linearLayer}.linear_attn.out_proj.weight`,
        `model.language_model.layers.${linearLayer}.linear_attn.conv1d.weight`,
        `model.language_model.layers.${linearLayer}.linear_attn.in_proj_a.weight`,
        `model.language_model.layers.${linearLayer}.linear_attn.in_proj_b.weight`,
      ]),
      [STABLE_LABEL_REFS.dense_ssm_ffn]: linesFromNames(model, [
        `model.language_model.layers.${linearLayer}.mlp.gate_proj.weight`,
        `model.language_model.layers.${linearLayer}.mlp.up_proj.weight`,
        `model.language_model.layers.${linearLayer}.mlp.down_proj.weight`,
      ]),
    },
    expectedTotal: listTensorNames(model)
      .filter(name => includeMtp || !name.startsWith('mtp.'))
      .reduce((total, name) => total + tensorCount(model, name), 0),
  };
}

function buildQwen35MoeEntry(model, presetId, includeMtp) {
  const layerIndices = getLayerIndicesForPrefix(model, 'model.language_model.layers');
  const selfLayer = layerIndices.find(index => hasPrefix(model, `model.language_model.layers.${index}.self_attn`));
  const linearLayer = layerIndices.find(index => hasPrefix(model, `model.language_model.layers.${index}.linear_attn`));
  const selfCount = layerIndices.filter(index => hasPrefix(model, `model.language_model.layers.${index}.self_attn`)).length;
  const linearCount = layerIndices.length - selfCount;
  const experts = getExpertCountFromRouter(model, `model.language_model.layers.${selfLayer}.mlp.gate.weight`);

  const z06 = [
    tensorLine(model, 'model.language_model.norm.weight'),
    scalarLineForPrefix(model, 'model.visual.'),
  ];
  if (includeMtp && hasPrefix(model, 'mtp.')) {
    z06.push(scalarLineForPrefix(model, 'mtp.'));
  }

  return {
    entry: {
      [STABLE_LABEL_REFS.moe_attention_layers]: String(selfCount),
      [STABLE_LABEL_REFS.moe_ssm_attention_layers]: String(linearCount),
      [STABLE_LABEL_REFS.embedding_shapes]: linesFromNames(model, [
        'model.language_model.embed_tokens.weight',
        'lm_head.weight',
      ]),
      [STABLE_LABEL_REFS.pre_first_norms]: z06.join('\n'),
      [STABLE_LABEL_REFS.experts_per_layer]: String(experts),
      [STABLE_LABEL_REFS.active_experts]: String(inferActiveExperts(model, presetId)),
      [STABLE_LABEL_REFS.shared_expert_scope]: 'per_layer',
      [STABLE_LABEL_REFS.shared_expert_tensors]: linesFromNames(model, [
        `model.language_model.layers.${selfLayer}.mlp.shared_expert.gate_proj.weight`,
        `model.language_model.layers.${selfLayer}.mlp.shared_expert.up_proj.weight`,
        `model.language_model.layers.${selfLayer}.mlp.shared_expert.down_proj.weight`,
      ]),
      [STABLE_LABEL_REFS.moe_attn]: linesFromNames(model, [
        `model.language_model.layers.${selfLayer}.self_attn.q_proj.weight`,
        `model.language_model.layers.${selfLayer}.self_attn.k_proj.weight`,
        `model.language_model.layers.${selfLayer}.self_attn.v_proj.weight`,
        `model.language_model.layers.${selfLayer}.self_attn.o_proj.weight`,
      ]),
      [STABLE_LABEL_REFS.moe_transitional]: linesFromNames(model, [
        `model.language_model.layers.${selfLayer}.input_layernorm.weight`,
        `model.language_model.layers.${selfLayer}.post_attention_layernorm.weight`,
        `model.language_model.layers.${selfLayer}.self_attn.q_norm.weight`,
        `model.language_model.layers.${selfLayer}.self_attn.k_norm.weight`,
      ]),
      [STABLE_LABEL_REFS.moe_shared_ffn]: linesFromNames(model, [
        `model.language_model.layers.${selfLayer}.mlp.gate.weight`,
        `model.language_model.layers.${selfLayer}.mlp.shared_expert_gate.weight`,
      ]),
      [STABLE_LABEL_REFS.moe_experts]: linesFromNames(model, [
        `model.language_model.layers.${selfLayer}.mlp.experts.gate_up_proj`,
        `model.language_model.layers.${selfLayer}.mlp.experts.down_proj`,
      ]),
      [STABLE_LABEL_REFS.moe_ssm_attn]: linesFromNames(model, [
        `model.language_model.layers.${linearLayer}.linear_attn.in_proj_qkv.weight`,
        `model.language_model.layers.${linearLayer}.linear_attn.in_proj_a.weight`,
        `model.language_model.layers.${linearLayer}.linear_attn.in_proj_b.weight`,
        `model.language_model.layers.${linearLayer}.linear_attn.in_proj_z.weight`,
        `model.language_model.layers.${linearLayer}.linear_attn.out_proj.weight`,
        `model.language_model.layers.${linearLayer}.linear_attn.conv1d.weight`,
      ]),
      [STABLE_LABEL_REFS.moe_ssm_transitional]: linesFromNames(model, [
        `model.language_model.layers.${linearLayer}.input_layernorm.weight`,
        `model.language_model.layers.${linearLayer}.post_attention_layernorm.weight`,
        `model.language_model.layers.${linearLayer}.linear_attn.norm.weight`,
        `model.language_model.layers.${linearLayer}.linear_attn.A_log`,
        `model.language_model.layers.${linearLayer}.linear_attn.dt_bias`,
      ]),
      [STABLE_LABEL_REFS.moe_ssm_shared_ffn]: linesFromNames(model, [
        `model.language_model.layers.${linearLayer}.mlp.gate.weight`,
        `model.language_model.layers.${linearLayer}.mlp.shared_expert_gate.weight`,
      ]),
      [STABLE_LABEL_REFS.moe_ssm_experts]: linesFromNames(model, [
        `model.language_model.layers.${linearLayer}.mlp.experts.gate_up_proj`,
        `model.language_model.layers.${linearLayer}.mlp.experts.down_proj`,
      ]),
      [STABLE_LABEL_REFS.has_shared_expert]: true,
      [STABLE_LABEL_REFS.experts_include_dim]: true,
    },
    expectedTotal: listTensorNames(model)
      .filter(name => includeMtp || !name.startsWith('mtp.'))
      .reduce((total, name) => total + tensorCount(model, name), 0),
  };
}

function buildMinimaxM2Entry(model, presetId) {
  const layerIndices = getLayerIndicesForPrefix(model, 'model.layers');
  const layer = layerIndices[0];
  const experts = getExpertCountFromRouter(model, `model.layers.${layer}.block_sparse_moe.gate.weight`);

  return {
    entry: {
      [STABLE_LABEL_REFS.moe_attention_layers]: String(layerIndices.length),
      [STABLE_LABEL_REFS.embedding_shapes]: linesFromNames(model, [
        'model.embed_tokens.weight',
        'lm_head.weight',
      ]),
      [STABLE_LABEL_REFS.pre_first_norms]: tensorLine(model, 'model.norm.weight'),
      [STABLE_LABEL_REFS.experts_per_layer]: String(experts),
      [STABLE_LABEL_REFS.active_experts]: String(model.architecture.num_experts_per_tok),
      [STABLE_LABEL_REFS.shared_expert_scope]: 'per_layer',
      [STABLE_LABEL_REFS.moe_attn]: linesFromNames(model, [
        `model.layers.${layer}.self_attn.q_proj.weight`,
        `model.layers.${layer}.self_attn.k_proj.weight`,
        `model.layers.${layer}.self_attn.v_proj.weight`,
        `model.layers.${layer}.self_attn.o_proj.weight`,
      ]),
      [STABLE_LABEL_REFS.moe_transitional]: linesFromNames(model, [
        `model.layers.${layer}.input_layernorm.weight`,
        `model.layers.${layer}.post_attention_layernorm.weight`,
        `model.layers.${layer}.self_attn.q_norm.weight`,
        `model.layers.${layer}.self_attn.k_norm.weight`,
        `model.layers.${layer}.self_attn.q_proj.weight_scale_inv`,
        `model.layers.${layer}.self_attn.k_proj.weight_scale_inv`,
        `model.layers.${layer}.self_attn.v_proj.weight_scale_inv`,
        `model.layers.${layer}.self_attn.o_proj.weight_scale_inv`,
      ]),
      [STABLE_LABEL_REFS.moe_shared_ffn]: linesFromNames(model, [
        `model.layers.${layer}.block_sparse_moe.gate.weight`,
        `model.layers.${layer}.block_sparse_moe.e_score_correction_bias`,
      ]),
      [STABLE_LABEL_REFS.moe_experts]: [
        expertLine(model, `model.layers.${layer}.block_sparse_moe.experts.0.w1.weight`, experts),
        expertLine(model, `model.layers.${layer}.block_sparse_moe.experts.0.w1.weight_scale_inv`, experts),
        expertLine(model, `model.layers.${layer}.block_sparse_moe.experts.0.w2.weight`, experts),
        expertLine(model, `model.layers.${layer}.block_sparse_moe.experts.0.w2.weight_scale_inv`, experts),
        expertLine(model, `model.layers.${layer}.block_sparse_moe.experts.0.w3.weight`, experts),
        expertLine(model, `model.layers.${layer}.block_sparse_moe.experts.0.w3.weight_scale_inv`, experts),
      ].join('\n'),
      [STABLE_LABEL_REFS.experts_include_dim]: true,
    },
  };
}

function buildGptOssEntry(model, presetId) {
  const layerIndices = getLayerIndicesForPrefix(model, 'model.layers');
  const layer = layerIndices[0];
  const experts = getExpertCountFromRouter(model, `model.layers.${layer}.mlp.router.weight`);
  const entry = {
    [STABLE_LABEL_REFS.moe_attention_layers]: String(layerIndices.length),
    [STABLE_LABEL_REFS.embedding_shapes]: linesFromExistingNames(model, [
      'model.embed_tokens.weight',
      'lm_head.weight',
    ]),
    [STABLE_LABEL_REFS.pre_first_norms]: tensorLine(model, 'model.norm.weight'),
    [STABLE_LABEL_REFS.experts_per_layer]: String(experts),
    [STABLE_LABEL_REFS.active_experts]: String(model.architecture.num_experts_per_tok),
    [STABLE_LABEL_REFS.shared_expert_scope]: 'per_layer',
    [STABLE_LABEL_REFS.moe_attn]: linesFromNames(model, [
      `model.layers.${layer}.self_attn.q_proj.weight`,
      `model.layers.${layer}.self_attn.q_proj.bias`,
      `model.layers.${layer}.self_attn.k_proj.weight`,
      `model.layers.${layer}.self_attn.k_proj.bias`,
      `model.layers.${layer}.self_attn.v_proj.weight`,
      `model.layers.${layer}.self_attn.v_proj.bias`,
      `model.layers.${layer}.self_attn.o_proj.weight`,
      `model.layers.${layer}.self_attn.o_proj.bias`,
      `model.layers.${layer}.self_attn.sinks`,
    ]),
    [STABLE_LABEL_REFS.moe_transitional]: linesFromNames(model, [
      `model.layers.${layer}.input_layernorm.weight`,
      `model.layers.${layer}.post_attention_layernorm.weight`,
    ]),
    [STABLE_LABEL_REFS.moe_shared_ffn]: linesFromNames(model, [
      `model.layers.${layer}.mlp.router.weight`,
      `model.layers.${layer}.mlp.router.bias`,
    ]),
    [STABLE_LABEL_REFS.moe_experts]: [
      shapeLine([experts, 5760, 2880]),
      shapeLine([experts, 2880, 2880]),
      tensorLine(model, `model.layers.${layer}.mlp.experts.gate_up_proj_bias`),
      tensorLine(model, `model.layers.${layer}.mlp.experts.down_proj_bias`),
    ].join('\n'),
    [STABLE_LABEL_REFS.experts_include_dim]: true,
  };

  return {
    entry,
    expectedTotal: computeResults(buildPresetInput(entry)).totalParams,
  };
}

function buildNemotronEntry(model, presetId, includeMtp) {
  const attnLayer = 7;
  const ssmLayer = 0;
  const moeSsmLayer = 1;
  const baseExpectedTotal = listTensorNames(model)
    .filter(name => !name.startsWith('mtp.'))
    .reduce((total, name) => total + tensorCount(model, name), 0);
  const mtpExpectedTotal = listTensorNames(model)
    .reduce((total, name) => total + tensorCount(model, name), 0);

  const z06 = [tensorLine(model, 'backbone.norm_f.weight')];
  if (includeMtp) {
    z06.push(nonEmptyJoin([
      linesFromNames(model, [
      'mtp.layers.0.eh_proj.weight',
      'mtp.layers.0.enorm.weight',
      'mtp.layers.0.hnorm.weight',
      'mtp.layers.0.norm.weight',
      'mtp.layers.0.mixer.q_proj.weight',
      'mtp.layers.0.mixer.k_proj.weight',
      'mtp.layers.0.mixer.v_proj.weight',
      'mtp.layers.0.mixer.o_proj.weight',
      'mtp.layers.1.final_layernorm.weight',
      'mtp.layers.1.norm.weight',
      'mtp.layers.1.mixer.fc1_latent_proj.weight',
      'mtp.layers.1.mixer.fc2_latent_proj.weight',
      'mtp.layers.1.mixer.gate.weight',
      'mtp.layers.1.mixer.gate.e_score_correction_bias',
      'mtp.layers.1.mixer.shared_experts.up_proj.weight',
      'mtp.layers.1.mixer.shared_experts.down_proj.weight',
      ]),
      shapeLine([512, ...requireTensor(model, 'mtp.layers.1.mixer.experts.0.up_proj.weight').shape]),
      shapeLine([512, ...requireTensor(model, 'mtp.layers.1.mixer.experts.0.down_proj.weight').shape]),
    ]));
  }

  return {
    entry: {
      [STABLE_LABEL_REFS.dense_attention_layers]: '8',
      [STABLE_LABEL_REFS.dense_ssm_attention_layers]: '40',
      [STABLE_LABEL_REFS.moe_ssm_attention_layers]: '40',
      [STABLE_LABEL_REFS.embedding_shapes]: linesFromNames(model, [
        'backbone.embeddings.weight',
        'lm_head.weight',
      ]),
      [STABLE_LABEL_REFS.pre_first_norms]: z06.join('\n'),
      [STABLE_LABEL_REFS.dense_norms]: tensorLine(model, `backbone.layers.${attnLayer}.norm.weight`),
      [STABLE_LABEL_REFS.dense_attn]: linesFromNames(model, [
        `backbone.layers.${attnLayer}.mixer.q_proj.weight`,
        `backbone.layers.${attnLayer}.mixer.k_proj.weight`,
        `backbone.layers.${attnLayer}.mixer.v_proj.weight`,
        `backbone.layers.${attnLayer}.mixer.o_proj.weight`,
      ]),
      [STABLE_LABEL_REFS.dense_ssm_norms]: tensorLine(model, `backbone.layers.${ssmLayer}.norm.weight`),
      [STABLE_LABEL_REFS.dense_ssm_attn]: linesFromNames(model, [
        `backbone.layers.${ssmLayer}.mixer.in_proj.weight`,
        `backbone.layers.${ssmLayer}.mixer.norm.weight`,
        `backbone.layers.${ssmLayer}.mixer.out_proj.weight`,
        `backbone.layers.${ssmLayer}.mixer.conv1d.weight`,
        `backbone.layers.${ssmLayer}.mixer.conv1d.bias`,
        `backbone.layers.${ssmLayer}.mixer.A_log`,
        `backbone.layers.${ssmLayer}.mixer.D`,
        `backbone.layers.${ssmLayer}.mixer.dt_bias`,
      ]),
      [STABLE_LABEL_REFS.experts_per_layer]: '512',
      [STABLE_LABEL_REFS.active_experts]: '22',
      [STABLE_LABEL_REFS.shared_expert_scope]: 'per_layer',
      [STABLE_LABEL_REFS.moe_ssm_transitional]: tensorLine(model, `backbone.layers.${moeSsmLayer}.norm.weight`),
      [STABLE_LABEL_REFS.moe_ssm_attn]: linesFromNames(model, [
        `backbone.layers.${moeSsmLayer}.mixer.fc1_latent_proj.weight`,
        `backbone.layers.${moeSsmLayer}.mixer.fc2_latent_proj.weight`,
      ]),
      [STABLE_LABEL_REFS.moe_ssm_shared_ffn]: linesFromNames(model, [
        `backbone.layers.${moeSsmLayer}.mixer.gate.weight`,
        `backbone.layers.${moeSsmLayer}.mixer.gate.e_score_correction_bias`,
        `backbone.layers.${moeSsmLayer}.mixer.shared_experts.up_proj.weight`,
        `backbone.layers.${moeSsmLayer}.mixer.shared_experts.down_proj.weight`,
      ]),
      [STABLE_LABEL_REFS.moe_ssm_experts]: [
        expertLine(model, `backbone.layers.${moeSsmLayer}.mixer.experts.0.up_proj.weight`, 512),
        expertLine(model, `backbone.layers.${moeSsmLayer}.mixer.experts.0.down_proj.weight`, 512),
      ].join('\n'),
      [STABLE_LABEL_REFS.has_shared_expert]: true,
      [STABLE_LABEL_REFS.experts_include_dim]: true,
    },
    expectedTotal: includeMtp ? mtpExpectedTotal : baseExpectedTotal,
  };
}

const ARCHITECTURE_BUILDERS = {
  kimi_k2: model => buildEntriesFromPresetExports(model, (presetId, includeMtp) => buildDeepseekStyleEntry(model, presetId, includeMtp)),
  deepseek_v3: model => buildEntriesFromPresetExports(model, (presetId, includeMtp) => buildDeepseekStyleEntry(model, presetId, includeMtp)),
  glm4_moe: model => buildEntriesFromPresetExports(model, (presetId, includeMtp) => buildGlm4MoeEntry(model, presetId, includeMtp)),
  glm4_moe_lite: model => buildEntriesFromPresetExports(model, (presetId, includeMtp) => buildGlm4MoeLiteEntry(model, presetId, includeMtp)),
  glm_moe_dsa: model => buildEntriesFromPresetExports(model, (presetId, includeMtp) => buildGlmMoeDsaEntry(model, presetId, includeMtp)),
  qwen3_moe: model => buildEntriesFromPresetExports(model, presetId => buildQwen3MoeEntry(model, presetId)),
  qwen3_5: model => buildEntriesFromPresetExports(model, (presetId, includeMtp) => buildQwen35DenseEntry(model, presetId, includeMtp)),
  qwen3_5_moe: model => buildEntriesFromPresetExports(model, (presetId, includeMtp) => buildQwen35MoeEntry(model, presetId, includeMtp)),
  minimax_m2: model => buildEntriesFromPresetExports(model, presetId => buildMinimaxM2Entry(model, presetId)),
  gpt_oss: model => buildEntriesFromPresetExports(model, presetId => buildGptOssEntry(model, presetId)),
  nemotron_h: model => buildEntriesFromPresetExports(model, (presetId, includeMtp) => buildNemotronEntry(model, presetId, includeMtp)),
};

function validatePreset(model, presetId, entry, explicitExpectedTotal) {
  const expectedTotal = explicitExpectedTotal ?? model.preset_exports?.[presetId]?.total_parameters;
  if (expectedTotal == null) {
    return;
  }

  const results = computeResults(buildPresetInput(entry));
  if (results.totalParams !== expectedTotal) {
    throw new Error(`Generated ${presetId} total ${results.totalParams} did not match expected ${expectedTotal}`);
  }
}

function main() {
  const inputPath = path.resolve(process.argv[2]);
  const outputPath = path.resolve(
    process.argv[3] || inputPath.replace(/\.json$/i, '.presets.json'),
  );

  if (!process.argv[2]) {
    console.error('Usage: node scripts/generate-presets-from-model.js <model-json> [output-json]');
    process.exit(1);
  }

  const model = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const modelType = model.architecture?.model_type;
  const builder = ARCHITECTURE_BUILDERS[modelType];

  if (!builder) {
    throw new Error(
      `Unsupported model architecture ${JSON.stringify(modelType)}. Add a builder to scripts/generate-presets-from-model.js.`,
    );
  }

  const { models, expectedTotals } = builder(model);
  const presets = {
    meta: {
      format: 'stable-ref-v1',
      source: `generated from ${path.relative(process.cwd(), inputPath)}`,
      notes: `Candidate preset entries derived from ${path.basename(inputPath)} without editing paramcalc.presets.json. Uses raw Hugging Face tensor orientations, so some display shapes may differ from hand-authored presets even when parameter totals match.`,
    },
    modelOrder: Object.keys(models),
    models,
  };

  for (const [presetId, entry] of Object.entries(models)) {
    validatePreset(model, presetId, entry, expectedTotals[presetId]);
  }

  fs.writeFileSync(outputPath, `${JSON.stringify(presets, null, 2)}\n`);
  process.stdout.write(`${path.relative(process.cwd(), outputPath)}\n`);
}

main();
