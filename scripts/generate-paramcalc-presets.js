const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const inputPath = path.join(root, 'paramcalc.presets.json');
const outputPath = path.join(root, 'paramcalc.presets.generated.js');

const presets = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
const banner = '// Generated from paramcalc.presets.json. Run `npm run build:paramcalc-presets` after editing presets.\n';
const body = `globalThis.PARAMCALC_PRESETS = ${JSON.stringify(presets, null, 2)};\n`;

fs.writeFileSync(outputPath, banner + body, 'utf8');
