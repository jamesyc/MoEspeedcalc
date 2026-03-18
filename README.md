# MoEspeedcalc

This repository contains a browser-based parameter calculator and the supporting preset-generation pipeline for dense, MoE, hybrid-attention, and SSM-style language models. The app itself lives in the HTML/CSS/JS files at the repo root, while the preset workflow starts from raw exported tensor-shape snapshots in `models/*.json`, generates candidate preset files in `models/*.presets.json`, syncs those candidates into `paramcalc.presets.json`, rebuilds the browser-consumable generated preset bundle, and verifies the result with the Node test suite.

`index.html` is the site entry point for the broader static page shell.

`paramcalc.html` is the main calculator UI markup for the parameter calculator itself.

`paramcalc.css` styles the calculator-specific layout, tables, and explanation output.

`paramcalc.js` contains the core parsing, preset-loading, computation, and rendering logic used by the calculator.

`paramcalc.presets.json` is the canonical preset source file consumed by the calculator and tests.

`paramcalc.presets.generated.js` is the generated browser bundle built from `paramcalc.presets.json`.

`paramcalc.test.js` is the Node test suite covering parsing, preset semantics, generated preset behavior, and stable refs.

`paramcalc.html` and `paramcalc.js` together define the stable `Z##` bucket model that the generated presets must satisfy.

`script.js` contains page-level client logic outside the calculator-specific code.

`style.css` contains site-wide styling outside the calculator-specific stylesheet.

`package.json` defines the small Node-based toolchain, including the preset build and test commands.

`README.md` is this root-level repo guide.

`models/` contains raw exported model tensor-shape JSON files and the generated candidate preset JSON files derived from them.

`scripts/` contains the utility scripts for exporting tensor shapes, generating candidate presets, syncing generated presets into the canonical preset file, and rebuilding the generated preset bundle.

## Workflow Path

1. Export raw tensor-shape snapshots into `models/*.json`.

2. Run [scripts/generate-presets-from-model.js](/Users/jameschang/git/MoEspeedcalc/scripts/generate-presets-from-model.js) over each model export to produce `models/*.presets.json`.

3. Review generated candidates in `models/` and compare them against the current canonical presets if needed.

4. Sync the generated candidate presets into [paramcalc.presets.json](/Users/jameschang/git/MoEspeedcalc/paramcalc.presets.json) with [scripts/sync-presets-from-models.js](/Users/jameschang/git/MoEspeedcalc/scripts/sync-presets-from-models.js).

5. Rebuild [paramcalc.presets.generated.js](/Users/jameschang/git/MoEspeedcalc/paramcalc.presets.generated.js) from the canonical JSON using `npm run build:paramcalc-presets`.

6. Run `npm test` to verify the preset file, generator assumptions, and calculator output semantics all remain consistent.
