#!/usr/bin/env node

const fs = require('node:fs');
const path = require('node:path');

function main() {
  const modelsDir = path.resolve(process.argv[2] || 'models');
  const outputPath = path.resolve(process.argv[3] || 'paramcalc.presets.json');

  const files = fs.readdirSync(modelsDir)
    .filter(name => name.endsWith('.presets.json'))
    .sort();

  const modelOrder = [];
  const models = {};

  for (const file of files) {
    const data = JSON.parse(fs.readFileSync(path.join(modelsDir, file), 'utf8'));
    for (const modelId of data.modelOrder || Object.keys(data.models || {})) {
      if (!data.models?.[modelId]) continue;
      modelOrder.push(modelId);
      models[modelId] = data.models[modelId];
    }
  }

  const merged = {
    meta: {
      format: 'stable-ref-v1',
      source: 'merged from models/*.presets.json',
      notes: 'Synchronized from generated raw-shape candidate preset files.',
    },
    modelOrder,
    models,
  };

  fs.writeFileSync(outputPath, `${JSON.stringify(merged, null, 2)}\n`);
  process.stdout.write(`${path.relative(process.cwd(), outputPath)}\n`);
}

main();
