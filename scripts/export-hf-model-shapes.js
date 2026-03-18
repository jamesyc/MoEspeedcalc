#!/usr/bin/env node

const fs = require('node:fs');
const path = require('node:path');

async function fetchJson(url) {
  const response = await fetch(url, {
    headers: {
      'user-agent': 'MoEspeedcalc model shape exporter',
    },
  });

  if (!response.ok) {
    throw new Error(`Request failed (${response.status}) for ${url}`);
  }

  return response.json();
}

async function fetchArrayBuffer(url, headers = {}) {
  const response = await fetch(url, {
    headers: {
      'user-agent': 'MoEspeedcalc model shape exporter',
      ...headers,
    },
  });

  if (!response.ok) {
    throw new Error(`Request failed (${response.status}) for ${url}`);
  }

  return response.arrayBuffer();
}

async function fetchShardHeader(repoId, revision, shardFile) {
  const shardUrl = `https://huggingface.co/${repoId}/resolve/${revision}/${encodeURIComponent(shardFile)}`;
  const lengthBuffer = Buffer.from(await fetchArrayBuffer(shardUrl, { Range: 'bytes=0-7' }));
  const headerLength = Number(lengthBuffer.readBigUInt64LE(0));
  const fullHeader = Buffer.from(
    await fetchArrayBuffer(shardUrl, { Range: `bytes=0-${headerLength + 7}` }),
  );
  const headerJson = fullHeader.subarray(8).toString('utf8');

  return JSON.parse(headerJson);
}

function summarizeLayerTensors(tensors, layerIndices) {
  const layerSet = new Set(layerIndices);
  let tensorCount = 0;
  let parameterCount = 0;

  for (const [name, tensor] of Object.entries(tensors)) {
    const match = name.match(/^model\.layers\.(\d+)\./);
    if (!match || !layerSet.has(Number(match[1]))) {
      continue;
    }
    tensorCount += 1;
    parameterCount += tensor.parameter_count;
  }

  return { tensorCount, parameterCount };
}

function buildGroups({ tensors, config }) {
  const layerIndices = [...new Set(
    Object.keys(tensors)
      .map((name) => {
        const match = name.match(/^model\.layers\.(\d+)\./);
        return match ? Number(match[1]) : null;
      })
      .filter((value) => value !== null),
  )].sort((a, b) => a - b);

  const sharedTensorNames = Object.keys(tensors)
    .filter((name) => !name.startsWith('model.layers.'))
    .sort();
  const sharedParameterCount = sharedTensorNames.reduce(
    (sum, name) => sum + tensors[name].parameter_count,
    0,
  );

  const denseLayerCount = Number(config.first_k_dense_replace || 0);
  const mtpLayerCount = Number(config.num_nextn_predict_layers || 0);
  const denseLayerIndices = layerIndices.filter((index) => index < denseLayerCount);
  const mtpLayerIndices = mtpLayerCount > 0 ? layerIndices.slice(-mtpLayerCount) : [];
  const mtpLayerSet = new Set(mtpLayerIndices);
  const moeLayerIndices = layerIndices.filter(
    (index) => index >= denseLayerCount && !mtpLayerSet.has(index),
  );

  const denseSummary = summarizeLayerTensors(tensors, denseLayerIndices);
  const moeSummary = summarizeLayerTensors(tensors, moeLayerIndices);
  const mtpSummary = summarizeLayerTensors(tensors, mtpLayerIndices);

  const mtpMarkerTensors = [];
  if (mtpLayerIndices.length > 0) {
    const referenceLayer = moeLayerIndices.at(-1);
    const referenceSuffixes = new Set(
      Object.keys(tensors)
        .filter((name) => name.startsWith(`model.layers.${referenceLayer}.`))
        .map((name) => name.slice(`model.layers.${referenceLayer}.`.length)),
    );

    for (const mtpLayerIndex of mtpLayerIndices) {
      const prefix = `model.layers.${mtpLayerIndex}.`;
      for (const tensorName of Object.keys(tensors).filter((name) => name.startsWith(prefix)).sort()) {
        const suffix = tensorName.slice(prefix.length);
        if (!referenceSuffixes.has(suffix)) {
          mtpMarkerTensors.push(tensorName);
        }
      }
    }
  }

  return {
    shared: {
      description: 'Tensors outside model.layers.* that are shared by both non-MTP and MTP exports.',
      tensor_names: sharedTensorNames,
      tensor_count: sharedTensorNames.length,
      parameter_count: sharedParameterCount,
    },
    dense_layers: {
      description: 'Early dense transformer layers before MoE routing begins.',
      layer_indices: denseLayerIndices,
      tensor_name_prefixes: denseLayerIndices.map((index) => `model.layers.${index}.`),
      tensor_count: denseSummary.tensorCount,
      parameter_count: denseSummary.parameterCount,
    },
    moe_layers: {
      description: 'Standard routed MoE layers included in both non-MTP and MTP exports.',
      layer_indices: moeLayerIndices,
      tensor_name_prefixes: moeLayerIndices.map((index) => `model.layers.${index}.`),
      tensor_count: moeSummary.tensorCount,
      parameter_count: moeSummary.parameterCount,
    },
    mtp_layers: {
      description: 'Next-token prediction layers that are only present in the MTP export.',
      layer_indices: mtpLayerIndices,
      tensor_name_prefixes: mtpLayerIndices.map((index) => `model.layers.${index}.`),
      marker_tensors: mtpMarkerTensors,
      tensor_count: mtpSummary.tensorCount,
      parameter_count: mtpSummary.parameterCount,
    },
  };
}

function buildPresetExports({ config, groups, totalParameters }) {
  const modelSlug = config._model_slug;
  const nonMtpId = modelSlug;
  const mtpId = `${modelSlug}-mtp`;
  const baseTotal =
    groups.shared.parameter_count +
    groups.dense_layers.parameter_count +
    groups.moe_layers.parameter_count;

  return {
    [nonMtpId]: {
      description: 'Export the non-MTP preset by excluding the MTP-only layer group.',
      include_groups: ['shared', 'dense_layers', 'moe_layers'],
      total_parameters: baseTotal,
    },
    [mtpId]: {
      description: 'Export the MTP preset by including the MTP-only layer group.',
      include_groups: ['shared', 'dense_layers', 'moe_layers', 'mtp_layers'],
      total_parameters: totalParameters,
      mtp_only_parameter_delta: groups.mtp_layers.parameter_count,
    },
  };
}

function buildModelDoc({ repoId, revision, indexJson, config, shardHeaders, repoUrl }) {
  const weightMap = indexJson.weight_map || {};
  const tensors = {};
  const shardTensorCounts = {};
  const shardTensorParamCounts = {};
  let totalParameters = 0;

  for (const [tensorName, shardFile] of Object.entries(weightMap)) {
    const header = shardHeaders.get(shardFile);
    const tensorMeta = header[tensorName];

    if (!tensorMeta) {
      throw new Error(`Missing tensor ${tensorName} in shard header ${shardFile}`);
    }

    const shape = Array.isArray(tensorMeta.shape) ? tensorMeta.shape.map(Number) : [];
    const parameterCount = shape.length === 0 ? 1 : shape.reduce((acc, dim) => acc * dim, 1);

    tensors[tensorName] = {
      dtype: tensorMeta.dtype,
      shape,
      shard: shardFile,
      data_offsets: tensorMeta.data_offsets,
      parameter_count: parameterCount,
    };

    shardTensorCounts[shardFile] = (shardTensorCounts[shardFile] || 0) + 1;
    shardTensorParamCounts[shardFile] = (shardTensorParamCounts[shardFile] || 0) + parameterCount;
    totalParameters += parameterCount;
  }

  const groups = buildGroups({ tensors, config });
  const presetExports = buildPresetExports({ config, groups, totalParameters });

  return {
    format: 'hf-safetensors-shapes-v1',
    model_id: repoId,
    revision,
    source: {
      repo_url: repoUrl,
      index_url: `https://huggingface.co/${repoId}/resolve/${revision}/model.safetensors.index.json`,
      config_url: `https://huggingface.co/${repoId}/resolve/${revision}/config.json`,
      extracted_at: new Date().toISOString(),
      total_size_bytes: indexJson.metadata?.total_size || null,
    },
    architecture: {
      model_type: config.model_type ?? null,
      hidden_size: config.hidden_size ?? null,
      intermediate_size: config.intermediate_size ?? null,
      moe_intermediate_size: config.moe_intermediate_size ?? null,
      num_hidden_layers: config.num_hidden_layers ?? null,
      first_k_dense_replace: config.first_k_dense_replace ?? null,
      num_nextn_predict_layers: config.num_nextn_predict_layers ?? null,
      n_routed_experts: config.n_routed_experts ?? null,
      n_shared_experts: config.n_shared_experts ?? null,
      num_experts_per_tok: config.num_experts_per_tok ?? null,
    },
    summary: {
      total_tensors: Object.keys(tensors).length,
      total_shards: shardHeaders.size,
      total_parameters: totalParameters,
    },
    groups,
    preset_exports: presetExports,
    shards: Object.keys(shardTensorCounts).sort().map((shardFile) => ({
      file: shardFile,
      tensor_count: shardTensorCounts[shardFile],
      parameter_count: shardTensorParamCounts[shardFile],
      metadata: shardHeaders.get(shardFile).__metadata__ || {},
    })),
    tensors,
  };
}

async function main() {
  const repoId = process.argv[2];
  const outputPath = process.argv[3];
  const revision = process.argv[4] || 'main';

  if (!repoId || !outputPath) {
    console.error('Usage: node scripts/export-hf-model-shapes.js <repo-id> <output-path> [revision]');
    process.exit(1);
  }

  const repoUrl = `https://huggingface.co/${repoId}/tree/${revision}`;
  const indexUrl = `https://huggingface.co/${repoId}/resolve/${revision}/model.safetensors.index.json`;
  const configUrl = `https://huggingface.co/${repoId}/resolve/${revision}/config.json`;
  const indexJson = await fetchJson(indexUrl);
  const config = await fetchJson(configUrl);
  config._model_slug = path.basename(outputPath, path.extname(outputPath));
  const uniqueShards = [...new Set(Object.values(indexJson.weight_map || {}))].sort();
  const shardHeaders = new Map();

  for (const shardFile of uniqueShards) {
    process.stderr.write(`Fetching ${shardFile}\n`);
    shardHeaders.set(shardFile, await fetchShardHeader(repoId, revision, shardFile));
  }

  const document = buildModelDoc({ repoId, revision, indexJson, config, shardHeaders, repoUrl });
  fs.mkdirSync(path.dirname(path.resolve(outputPath)), { recursive: true });
  fs.writeFileSync(path.resolve(outputPath), `${JSON.stringify(document, null, 2)}\n`);
  process.stderr.write(`Wrote ${outputPath}\n`);
}

main().catch((error) => {
  console.error(error.stack || String(error));
  process.exit(1);
});
