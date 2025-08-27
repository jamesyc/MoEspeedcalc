/*
 * Mixture‑of‑Experts Model Speed Calculator
 *
 * This script reads hardware and model parameters from the form,
 * computes intermediate values, and renders a table of results
 * complete with human‑readable formulas.  Calculations assume
 * bandwidth‑bound execution; compute capabilities are not factored in.
 */

// Helper: format a large number with commas as thousands separators
function formatNumber(n) {
  return n.toLocaleString('en-US');
}

// Helper: format a value to a fixed number of decimal places and remove
// trailing zeros.  If decimals = 0, returns an integer string.
function formatFloat(value, decimals) {
  const s = value.toFixed(decimals);
  return s.replace(/\.0+$/, '').replace(/(\.[0-9]*[1-9])0+$/, '$1');
}

// Prefill model fields based on selection
function prefillModel(model) {
  const totalParamsEl = document.getElementById('total_params');
  const denseParamsEl = document.getElementById('dense_params');
  const moeParamsEl = document.getElementById('moe_params');
  const kvCacheEl = document.getElementById('kv_cache');

  if (model === 'kimi') {
    // Kimi‑K2 preset parameters
    totalParamsEl.value = 1026470731056;
    denseParamsEl.value = 11722775856;
    moeParamsEl.value = 21140582400;
    // Do not override existing KV cache when switching models
  } else if (model === 'qwen') {
    // Qwen3‑235B preset parameters
    totalParamsEl.value = 235044351488;
    denseParamsEl.value = 7947951616;
    moeParamsEl.value = 14193524736;
    // Preserve KV cache value
  } else if (model === 'deepseek') {
    // Deepseek‑R1‑0528 preset parameters
    totalParamsEl.value = 671026419200;
    // Always‑active dense parameters per token
    denseParamsEl.value = 14563317248;
    // Active MoE parameters per token
    moeParamsEl.value = 22988980224;
    // Keep existing KV cache value
  } else if (model === 'gpt-oss-120b') {
    // gpt‑oss‑120b preset parameters
    totalParamsEl.value = 116829156672;
    denseParamsEl.value = 1548424512;
    moeParamsEl.value = 3584424960;
    // Keep existing KV cache value
  } else if (model === 'glm-4.5-air') {
    // GLM 4.5 Air preset parameters
    totalParamsEl.value = 106852251264;
    denseParamsEl.value = 6393421824;
    moeParamsEl.value = 7030707840;
    // Keep existing KV cache value
  } else {
    // Custom – clear fields including KV cache
    totalParamsEl.value = '';
    denseParamsEl.value = '';
    moeParamsEl.value = '';
    kvCacheEl.value = '';
  }
}

// GPU presets definition.  Each entry defines a friendly name along with
// memory capacity (in GB) and memory bandwidth (in GB/s).  Values are
// derived from manufacturer specifications and reputable sources.  For
// example, the Tesla P40 datasheet lists 24 GB of memory and 346 GB/s
// bandwidth【142303946954249†L37-L40】; the Tesla P100 datasheet lists 16 GB and 732 GB/s【701166213550239†L65-L69】; the Radeon
// Instinct MI25 article reports 16 GB HBM2 with 484 GB/s bandwidth【804931751416487†L120-L132】; Tom's Hardware
// notes that Radeon VII features 16 GB of HBM2 and a “massive 1 TB/s of
// memory bandwidth”【277597849462090†L236-L238】.  Intel’s Arc A770 specifications list a 16 GB memory size
// and 560 GB/s bandwidth【315001550906279†L212-L226】.  TechPowerUp entries provide memory
// bandwidths for MI60【581808855564426†L868-L870】, MI100【420210633025724†L255-L256】, MI210【555347255352234†L155-L166】 and
// MI250【977791413358372†L144-L154】.  The RTX 4060 Ti has 16 GB of GDDR6 and 288 GB/s bandwidth【334990342015257†L884-L897】.
// Presets for GPUs ordered by manufacturer (NVIDIA, AMD, Intel), then by
// segment (consumer, workstation, data center), and finally by release
// year (older to newer).  Each entry includes a friendly name along
// with memory capacity (GB) and memory bandwidth (GB/s).  Release
// years were determined from manufacturer announcements and GPU
// databases: for example, the GeForce RTX 3090 launched on September 1,
// 2020【798265537916332†L96-L103】 while the RTX 4090 debuted on
// September 20, 2022【189622123745852†L92-L99】.  AMD’s Radeon VII launched
// February 7, 2019【629533376336865†L96-L104】, the Radeon RX 7900 XTX
// launched on November 3, 2022【23499749435156†L96-L100】, and Intel’s Arc A770
// appeared on October 12, 2022【329656823451217†L84-L88】.  Data‑center
// accelerators such as the Tesla P40 and P100 were announced in 2016
//【233235703089481†L120-L167】, the A100 in 2020【522355814013881†L66-L70】, the H100 in
// 2022【226245728211610†L66-L68】 and the B200 in 2024【655953145353578†L64-L108】.  See
// script.js comments at top for bandwidth citations.
const gpuPresets = [
  { key: 'custom', name: 'Custom (enter your own values)', vram: null, bw: null },
  // NVIDIA consumer GPUs
  { key: 'rtx3090', name: 'NVIDIA GeForce RTX 3090 (24 GB, 936 GB/s)', vram: 24, bw: 936 },
  { key: 'rtx4060ti', name: 'NVIDIA GeForce RTX 4060 Ti (16 GB, 288 GB/s)', vram: 16, bw: 288 },
  { key: 'rtx4070', name: 'NVIDIA GeForce RTX 4070 (12 GB, 504 GB/s)', vram: 12, bw: 504 },
  { key: 'rtx4090', name: 'NVIDIA GeForce RTX 4090 (24 GB, 1008 GB/s)', vram: 24, bw: 1008 },
  { key: 'rtx5090', name: 'NVIDIA GeForce RTX 5090 (32 GB, 1792 GB/s)', vram: 32, bw: 1792 },
  // NVIDIA workstation GPUs
  { key: 'a6000', name: 'NVIDIA RTX A6000 (48 GB, 768 GB/s)', vram: 48, bw: 768 },
  { key: 'a5000', name: 'NVIDIA RTX A5000 (24 GB, 768 GB/s)', vram: 24, bw: 768 },
  { key: 'rtxpro6000', name: 'NVIDIA RTX Pro 6000 Blackwell (96 GB, 1792 GB/s)', vram: 96, bw: 1792 },
  // NVIDIA datacenter GPUs
  { key: 'p40', name: 'NVIDIA P40 (24 GB, 346 GB/s)', vram: 24, bw: 346 },
  { key: 'p100', name: 'NVIDIA P100 (16 GB, 732 GB/s)', vram: 16, bw: 732 },
  { key: 'a100', name: 'NVIDIA A100 80GB (80 GB, 1940 GB/s)', vram: 80, bw: 1940 },
  { key: 'h100', name: 'NVIDIA H100 80GB (80 GB, 2000 GB/s)', vram: 80, bw: 2000 },
  { key: 'b200', name: 'NVIDIA B200 (192 GB, 8200 GB/s)', vram: 192, bw: 8200 },
  // AMD consumer GPUs
  { key: 'radeonvii', name: 'AMD Radeon VII (16 GB, 1024 GB/s)', vram: 16, bw: 1024 },
  { key: 'rx7900xtx', name: 'AMD Radeon RX 7900 XTX (24 GB, 960 GB/s)', vram: 24, bw: 960 },
  // AMD datacenter GPUs
  { key: 'mi25', name: 'AMD Instinct MI25 (16 GB, 484 GB/s)', vram: 16, bw: 484 },
  { key: 'mi50', name: 'AMD Instinct MI50 (32 GB, 1024 GB/s)', vram: 32, bw: 1024 },
  { key: 'mi60', name: 'AMD Instinct MI60 (32 GB, 1024 GB/s)', vram: 32, bw: 1024 },
  { key: 'mi100', name: 'AMD Instinct MI100 (32 GB, 1230 GB/s)', vram: 32, bw: 1230 },
  { key: 'mi210', name: 'AMD Instinct MI210 (64 GB, 1640 GB/s)', vram: 64, bw: 1640 },
  { key: 'mi250', name: 'AMD Instinct MI250 (128 GB, 3280 GB/s)', vram: 128, bw: 3280 },
  // Intel consumer GPU
  { key: 'arca770', name: 'Intel Arc A770 (16 GB, 560 GB/s)', vram: 16, bw: 560 }
];

// Populate a GPU select element with preset options
function populateGpuSelect(selectId) {
  const select = document.getElementById(selectId);
  gpuPresets.forEach(preset => {
    const opt = document.createElement('option');
    opt.value = preset.key;
    opt.textContent = preset.name;
    select.appendChild(opt);
  });
}

// Prefill GPU VRAM and bandwidth fields when a preset is selected
function handleGpuPresetChange(selectId, vramInputId, bwInputId) {
  const select = document.getElementById(selectId);
  const vramInput = document.getElementById(vramInputId);
  const bwInput = document.getElementById(bwInputId);
  select.addEventListener('change', () => {
    const selectedKey = select.value;
    const preset = gpuPresets.find(p => p.key === selectedKey);
    if (!preset) return;
    // Only override values when a preset other than custom is chosen
    if (preset.key !== 'custom') {
      vramInput.value = preset.vram;
      bwInput.value = preset.bw;
    }
    // For custom, leave existing values unchanged
  });
}

// Perform the calculations and render results
function calculate() {
  const resultsDiv = document.getElementById('results');
  // Gather hardware parameters
  const gpu1Vram = parseFloat(document.getElementById('gpu1_vram').value);
  const gpu1Bw = parseFloat(document.getElementById('gpu1_bw').value);
  const gpu2Vram = parseFloat(document.getElementById('gpu2_vram').value);
  const gpu2Bw = parseFloat(document.getElementById('gpu2_bw').value);
  const systemBw = parseFloat(document.getElementById('system_bw').value);
  // Gather model parameters
  const totalParams = parseFloat(document.getElementById('total_params').value);
  const denseParams = parseFloat(document.getElementById('dense_params').value);
  const activeMoeParams = parseFloat(document.getElementById('moe_params').value);
  const kvCache = parseFloat(document.getElementById('kv_cache').value) || 0;
  const quantBits = parseInt(document.getElementById('quantization').value);

  // Validate required fields; if any are missing or invalid, abort
  if ([gpu1Vram, gpu1Bw, gpu2Vram, gpu2Bw, systemBw, totalParams, denseParams, activeMoeParams].some(v => isNaN(v) || v < 0)) {
    resultsDiv.innerHTML = '<div class="info-text">Please fill in all fields with valid (non‑negative) numbers before calculating.</div>';
    resultsDiv.classList.remove('hidden');
    return;
  }

  // Derived values
  const totalMoeParams = Math.max(totalParams - denseParams, 0);
  const bytesPerParam = quantBits / 8.0;

  // Dense part size in GB = dense params * bytes per param / 1e9
  const denseSizeGB = denseParams * bytesPerParam / 1e9;
  const denseKvGB = denseSizeGB + kvCache;
  const fitsOnGpu1 = denseKvGB <= gpu1Vram;

  // Time to load dense+kv (ms) = size (GB) / bandwidth (GB/s) * 1000
  const denseLoadMs = gpu1Bw > 0 ? (denseKvGB / gpu1Bw) * 1000 : Infinity;

  // Total MoE size in GB
  const moeTotalGB = totalMoeParams * bytesPerParam / 1e9;
  // Share of MoE stored on GPU2
  let moeShare = 0;
  if (moeTotalGB > 0 && gpu2Vram > 0) {
    moeShare = Math.min(1, gpu2Vram / moeTotalGB);
  }
  // Time for MoE portion on GPU2
  const activeMoeSizeGB = activeMoeParams * bytesPerParam / 1e9;
  const gpu2MoeMs = gpu2Bw > 0 ? (moeShare * activeMoeSizeGB / gpu2Bw) * 1000 : Infinity;
  // Time for MoE portion from system RAM
  const systemShare = 1 - moeShare;
  const systemMoeMs = systemBw > 0 ? (systemShare * activeMoeSizeGB / systemBw) * 1000 : Infinity;

  // Total time per token (ms) and throughput (tokens/s)
  const totalMsPerToken = denseLoadMs + gpu2MoeMs + systemMoeMs;
  const tokensPerSec = totalMsPerToken > 0 ? (1000 / totalMsPerToken) : 0;

  // Build HTML output with formulas and values
  let html = '';
  html += '<h2>Results</h2>';
  // Dense + KV size
  html += '<div class="result-row">';
  html += '<span class="result-title">Dense parameters + KV cache size:</span>';
  html += `<span class="result-value">${formatFloat(denseKvGB, 4)}&nbsp;GB</span>`;
  // Show units for dense size calculation.  denseParams has units of parameters, quantBits is bits/param.
  const eq1 = `${formatNumber(denseParams)} params × (${quantBits} bits/param ÷ 8 bits/byte) ÷ 1e9 bytes/GB + ${kvCache} GB = ${formatFloat(denseKvGB, 4)} GB`;
  html += `<div class="equation">${eq1}</div>`;
  html += '</div>';
  // Fit in GPU1
  html += '<div class="result-row">';
  html += '<span class="result-title">Fits in GPU&nbsp;1 memory?</span>';
  html += `<span class="result-value">${fitsOnGpu1 ? 'Yes' : 'No'}</span>`;
  // Indicate units when checking memory fit (GB).
  const eqFit = `${formatFloat(denseKvGB, 4)} GB ≤ ${gpu1Vram} GB → ${fitsOnGpu1 ? 'True' : 'False'}`;
  html += `<div class="equation">${eqFit}</div>`;
  html += '</div>';
  // Dense+KV load time
  html += '<div class="result-row">';
  html += '<span class="result-title">Dense/KV load time:</span>';
  html += `<span class="result-value">${formatFloat(denseLoadMs, 3)}&nbsp;ms</span>`;
  // For load time, divide GB by GB/s then multiply by 1000 ms/s.
  const eq2 = `${formatFloat(denseKvGB, 4)} GB ÷ ${gpu1Bw} GB/s × 1000 ms/s = ${formatFloat(denseLoadMs, 3)} ms`;
  html += `<div class="equation">${eq2}</div>`;
  html += '</div>';
  // MoE total size
  html += '<div class="result-row">';
  html += '<span class="result-title">Total MoE size:</span>';
  html += `<span class="result-value">${formatFloat(moeTotalGB, 4)}&nbsp;GB</span>`;
  // Total MoE size calculation with units: params × bits/param ÷ (8 bits/byte) ÷ 1e9 bytes/GB.
  const eq3 = `${formatNumber(totalMoeParams)} params × (${quantBits} bits/param ÷ 8 bits/byte) ÷ 1e9 bytes/GB = ${formatFloat(moeTotalGB, 4)} GB`;
  html += `<div class="equation">${eq3}</div>`;
  html += '</div>';
  // GPU2 share
  html += '<div class="result-row">';
  html += '<span class="result-title">Percentage of MoE on GPU&nbsp;2:</span>';
  html += `<span class="result-value">${formatFloat(moeShare * 100, 2)}&nbsp;%</span>`;
  // GPU2 share calculation shows GB units cancelling out.
  const eq4 = `min(1, ${gpu2Vram} GB ÷ ${formatFloat(moeTotalGB, 4)} GB) = ${formatFloat(moeShare, 4)}`;
  html += `<div class="equation">${eq4}</div>`;
  html += '</div>';
  // GPU2 MoE time
  html += '<div class="result-row">';
  html += '<span class="result-title">MoE load time on GPU&nbsp;2:</span>';
  html += `<span class="result-value">${formatFloat(gpu2MoeMs, 3)}&nbsp;ms</span>`;
  // GPU2 MoE load time: share (unitless) × activeMoeParams × bits/param ÷ (8 bits/byte) ÷ 1e9 bytes/GB ÷ (GPU2 GB/s) × 1000 ms/s.
  const eq5 = `${formatFloat(moeShare, 4)} × (${formatNumber(activeMoeParams)} params × (${quantBits} bits/param ÷ 8 bits/byte) ÷ 1e9 bytes/GB) ÷ ${gpu2Bw} GB/s × 1000 ms/s = ${formatFloat(gpu2MoeMs, 3)} ms`;
  html += `<div class="equation">${eq5}</div>`;
  html += '</div>';
  // System RAM time
  html += '<div class="result-row">';
  html += '<span class="result-title">MoE load time from system RAM:</span>';
  html += `<span class="result-value">${formatFloat(systemMoeMs, 3)}&nbsp;ms</span>`;
  // System RAM MoE load time: similar to GPU2 but using system bandwidth.
  const eq6 = `${formatFloat(systemShare, 4)} × (${formatNumber(activeMoeParams)} params × (${quantBits} bits/param ÷ 8 bits/byte) ÷ 1e9 bytes/GB) ÷ ${systemBw} GB/s × 1000 ms/s = ${formatFloat(systemMoeMs, 3)} ms`;
  html += `<div class="equation">${eq6}</div>`;
  html += '</div>';
  // Total ms per token
  html += '<div class="result-row">';
  html += '<span class="result-title">Total time per token:</span>';
  html += `<span class="result-value">${formatFloat(totalMsPerToken, 3)}&nbsp;ms</span>`;
  // Summation of component times with ms units.
  const eq7 = `${formatFloat(denseLoadMs, 3)} ms + ${formatFloat(gpu2MoeMs, 3)} ms + ${formatFloat(systemMoeMs, 3)} ms = ${formatFloat(totalMsPerToken, 3)} ms`;
  html += `<div class="equation">${eq7}</div>`;
  html += '</div>';
  // Tokens per second
  html += '<div class="result-row">';
  html += '<span class="result-title">Tokens per second:</span>';
  html += `<span class="result-value">${formatFloat(tokensPerSec, 2)}&nbsp;tokens/s</span>`;
  // Tokens per second: 1000 ms/s divided by ms per token yields tokens/s.
  const eq8 = `1000 ms/s ÷ ${formatFloat(totalMsPerToken, 3)} ms = ${formatFloat(tokensPerSec, 2)} tokens/s`;
  html += `<div class="equation">${eq8}</div>`;
  html += '</div>';

  resultsDiv.innerHTML = html;
  resultsDiv.classList.remove('hidden');

  // After rendering the results, scroll smoothly to the results section so the user
  // can immediately see the output.  The scrollIntoView API provides a
  // convenient way to move the viewport.  Use smooth behavior for a pleasant
  // transition.
  if (typeof resultsDiv.scrollIntoView === 'function') {
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

// Initialize event listeners once DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  // Prefill when model selection changes
  const modelSelect = document.getElementById('model-select');
  modelSelect.addEventListener('change', () => {
    prefillModel(modelSelect.value);
  });
  // Initially, ensure custom fields are blank
  prefillModel('custom');

  // Populate GPU preset selectors and add change handlers
  populateGpuSelect('gpu1-select');
  populateGpuSelect('gpu2-select');
  // Set default selection to custom
  document.getElementById('gpu1-select').value = 'custom';
  document.getElementById('gpu2-select').value = 'custom';
  // Attach change handlers to override VRAM/bandwidth fields when preset selected
  handleGpuPresetChange('gpu1-select', 'gpu1_vram', 'gpu1_bw');
  handleGpuPresetChange('gpu2-select', 'gpu2_vram', 'gpu2_bw');
  // Bind calculate button
  document.getElementById('calculate-btn').addEventListener('click', (e) => {
    e.preventDefault();
    calculate();
  });
});