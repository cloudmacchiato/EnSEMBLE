# ENSEMBLE

ENSEMBLE provides enhancer-centric differential analysis and enrichment workflows.

## Enhancer Set Enrichment Analysis (ESEA)

- To reproduce the bundled workflow, run `Rscript run_k562_example.R [optional_output_dir]`. When no directory is supplied, results are written to `outputs/k562_example` under the current working directory.
- The script automates loading the package, filters enhancer sets via `retain_specific_enhancers()`, and saves the aligned metadata plus ESEA and GSEA tables inside the output directory.
- Ensure the file paths defined near the top of `run_k562_example.R` point to your local copies of the K562 count matrices, metadata, and MSigDB collections before launching the script.

### Synthetic TNBC demo

A lightweight SNAI1 knockout TNBC dataset now ships with the package under
`inst/extdata/example_data`. Use `ensemble_example_data()` to locate the files
inside an installed copy of ENSEMBLE or call `use_example_data()` to copy the
assets to a writable directory:

```r
dest <- use_example_data()
list.files(dest)
#> example_background.txt, example_metadata.csv, example_enhancer_counts.tsv, ...
```

The folder contains enhancer and gene count matrices, metadata, helper/ESEA/GSEA
tables, minimalist GeneHancer and GTF annotations, enhancer-set GMTs, and a
filled background form. Point the file paths in `run_k562_example.R` (or your
own workflow) to these files to execute the entire pipeline end-to-end without
downloading external resources.

## Enhancer Overlap Enrichment Analysis (eORA)

The eORA utilities map SNPs or loci to GeneHancer IDs and test enrichment
against enhancer sets. The GeneHancer BED is assumed to be BED0 (0-based start,
end-exclusive) and is converted internally to 1-based coordinates for
`GenomicRanges`.

Example: map rsIDs to GHIDs from the bundled demo inputs.

```r
source("R/eORA.R")
snps_in <- readLines("inst/extdata/example_eORA_SNPs.txt")
enhancer_sets <- read_concepts("normal_CellType2Enhancer_v2.gmt")
res <- run_eORA(
    snps = snps_in,
    gh_bed = "inst/extdata/GeneHancer_v5.24.bed",
    enhancer_sets = enhancer_sets,
    ghid_col = 4,
    B = 10000,
    seed = 1,
    alpha = 0.05,
    p_adjust_method = "BH",
    input_type = "rsid"
)
```

Notes:
- `run_eORA` accepts rsIDs, locus strings (`chr:pos` or `chr:start-end`), or
  data frames with `chr+pos` or `chr+start+end`.
- rsID mapping uses `biomaRt` and requires network access; use locus inputs if
  you need fully offline runs. If biomaRt returns 0-based positions, the code
  shifts to 1-based and emits a message when `quiet = FALSE`.

## Installation

ENSEMBLE depends on several Bioconductor packages (`DESeq2`, `edgeR`,
`fgsea`, `GenomicRanges`, `IRanges`, `S4Vectors`, `rtracklayer`) plus CRAN
packages such as `data.table`, `Matrix`, and `jsonlite`. Install the required
Bioconductor components first, then pull ENSEMBLE from GitHub:

```r
install.packages("BiocManager")
BiocManager::install(c("DESeq2", "edgeR", "fgsea", "GenomicRanges",
                       "IRanges", "S4Vectors", "rtracklayer"))

install.packages("devtools")
devtools::install_github("cloudmacchiato/ENSEMBLE")
```

After installation, attach the package and locate the bundled demo files:

```r
library(ENSEMBLE)
ensemble_example_data()
```

### Swapping in your own datasets

`run_k562_example.R` now reads the bundled example paths by default and lets you
override any input via `example_config.json` in your working directory. Update
the JSON with lab-specific files (enhancer/gene counts, metadata, annotations,
MSigDB GMTs, and contrast definition) and rerun the script:

```json
{
  "counts_file_enh": "/path/to/GeneHancer_counts.tsv",
  "counts_file_gene": "/path/to/HGNC_counts.tsv",
  "genehancer_file": "/path/to/GeneHancer.saf",
  "gencode_file": "/path/to/gencode.gtf",
  "enhancer_set_file": "/path/to/enhancer_sets.gmt",
  "metadata_file": "/path/to/metadata.csv",
  "msigdb_files": ["/path/msigdb/h.all.v2024.1.Hs.symbols.gmt"],
  "contrast": ["group", "Treatment", "Control"]
}
```

Leave keys untouched to keep using the packaged TNBC demo.

# Gemini ESEA↔GSEA Agent Guide

This guide walks through using the Gemini-backed agent end-to-end: creating an isolated environment, configuring API credentials, and running the workflow on your enhancer and pathway tables.

## 1. Prerequisites
- Python 3.9 or newer (3.10 recommended)
- `conda` (optional but recommended for isolation)
- Access to the [Google AI Studio](https://ai.google.dev/) Gemini API
- Input files: a GSEA CSV, an ESEA CSV, and a completed background form from this repository

## 2. Create a Dedicated Environment
Use the provided Conda environment or create your own:

```bash
# Create a fresh environment
conda create -n ensemble python=3.10
conda activate ensemble
```

Install the Python requirements (WeasyPrint needs system libraries such as
`libpango-1.0-0`, `libcairo2`, `libgdk-pixbuf-2.0-0`, and `libffi-dev` on
Debian/Ubuntu):

```bash
pip install -r requirements.txt
```

> On macOS, install `pango`, `cairo`, and `gdk-pixbuf` via Homebrew
> (`brew install pango cairo gdk-pixbuf`) before running `pip install`.

The `requirements.txt` file pins `google-generativeai` to the 0.5–0.6 range
tested with this agent; upgrade only after verifying the pipeline locally.

## 3. Configure Your Gemini API Key
1. Generate an API key from Google AI Studio and store it in a safe place **outside version control**. For example:
   ```bash
   echo "YOUR_KEY_HERE" > gemini_api_key.txt
   chmod 600 gemini_api_key.txt
   ```
2. You can provide the key in one of two ways when running the agent:
   - Export the environment variable once per shell session:
     ```bash
     export GOOGLE_API_KEY=$(cat gemini_api_key.txt)
     ```
   - Or pass it via the CLI flag:
     ```bash
     --gemini-api-key $(cat gemini_api_key.txt)
     ```

## 4. Run the Agent
The CLI only needs paths to your data plus the Gemini options; the Gemini backend is now the sole engine. Example using the shared `singlecell` environment and an inline key flag:

```bash
python -m local_agent.cli \
  --gsea-csv GSEA_results.csv \
  --esea-csv ESEA_results.csv \
  --background-txt background_k562_example.txt \
  --output-dir outputs/gemini_run \
  --gemini-api-key $(cat gemini_api_key.txt)
```

Note: keep a trailing `\` on every intermediate line (or put the command on a single line). If a backslash is missing the shell treats the next `--flag` as a new command and prints `command not found`. Leading-edge genes for each top pathway are retained (trimmed to the most informative 10 genes), so the agent weighs those driver genes during drafting and verification. Enhancer tables drop their massive leading-edge lists to keep room for ~10 cell types per direction. Ensure your background file includes a `Contrast:` line describing numerator vs reference so NES/effect signs are interpreted correctly. When you add `--gsea-only`, the run completes without enhancer evidence and the outputs highlight that follow-up step.

## 4.1 CLI Parameters
- `--gsea-csv PATH` (required): Absolute or relative path to the Gene Set Enrichment results CSV. The pipeline keeps only rows that pass `--gsea-q-threshold` and `--gsea-top-n`.
- `--esea-csv PATH` (optional): Path to the enhancer screen CSV. Provide it for enhancer-aware verification; omit when running in GSEA-only mode.
- `--background-txt PATH` (required): Plain-text form describing the study context. Use `background_form_template.txt` as the scaffold.
- `--output-dir PATH` (default `outputs/gemini_agent`): Directory where artefacts such as `mini_thesis.txt` and `verification_table.txt` are written.
- `--gemini-model NAME` (default `models/gemini-2.5-flash`): Gemini model identifier. Lower-capacity models (e.g., `models/gemini-1.5-flash`) consume fewer tokens if you are rate-limited.
- `--gemini-api-key KEY` (optional): Inline Gemini API key. Skip when `GOOGLE_API_KEY` is already exported in the shell.
- `--gsea-only` (flag): Skip enhancer (ESEA) inputs and treat all findings as pathway-driven hypotheses.
- `--gemini-temperature FLOAT` (default `0.3`): Sampling temperature. Lower values make the prose more deterministic; higher values add diversity.
- `--gemini-top-p FLOAT` (default `0.95`): Nucleus sampling cutoff. Reduce to tighten the probability mass considered at each token.
- `--gemini-top-k INT` (default `32`): Limits how many candidates are inspected during sampling. Smaller numbers reduce randomness slightly.
- `--gemini-max-output-tokens INT` (default `8192`): Hard ceiling on the response length. Increase if long summaries are truncated, at the cost of more quota usage.
- `--gsea-top-n INT` (default `50`): Maximum pathway records per direction (up/down) forwarded to Gemini after filtering.
- `--gsea-q-threshold FLOAT` (default `0.05`): Q-value cutoff for GSEA pathways. Lower this to send only the most confident pathways.
- `--theme-cap INT` (default `3`): Cap on the number of themed pathway groups per direction that are described to Gemini.
- `--esea-max-per-direction INT` (default `10`): Maximum enhancer cell types per direction forwarded to Gemini; guarantees at least one UP and one DOWN candidate when available.
- `--esea-q-threshold FLOAT` (default `0.05`): Q-value cutoff for enhancer hits to count as “Supported”.
- `--esea-effect-threshold FLOAT` (default `0.30`): Minimum absolute effect size to count as “Supported”.
- `--esea-partial-q-threshold FLOAT` (default `0.10`): Q-value limit for classifying a hit as “Partial”.
- `--esea-partial-effect-threshold FLOAT` (default `0.20`): Effect-size threshold for the “Partial” decision.

## 4.2 Generation Tuning Tips
- The default model (`models/gemini-2.5-flash`) balances quality and latency but you can swap to lighter flashes if quotas remain tight. Swap `--gemini-model` for another identifier returned by `genai.list_models()` if desired.
- Tune generation behaviour with `--gemini-temperature`, `--gemini-top-p`, `--gemini-top-k`, and `--gemini-max-output-tokens` (defaults to 0.3 / 0.95 / 32 / 8192 respectively).
- Control deterministic filtering with the new flags:
  - `--gsea-top-n` (default 50) and `--gsea-q-threshold` (default 0.05) limit which pathway themes are shown to Gemini.
  - `--theme-cap` (default 3) caps the number of theme groups per direction.
  - `--esea-max-per-direction` (default 10) keeps roughly ten enhancer states per sign while still allowing balanced coverage.
  - `--esea-q-threshold` (default 0.05) and `--esea-effect-threshold` (default 0.30) gate supported enhancer hits.
  - `--esea-partial-q-threshold` (default 0.10) and `--esea-partial-effect-threshold` (default 0.20) adjust when a claim is downgraded to “Partial”.

## 4.3 Background Contrast Checklist
- Always fill the `Contrast:` field in `background_form_template.txt` (e.g., `Treatment vs Control`).
- Spell out which cohort is the numerator so the agent maps positive/negative NES correctly.
- If directionality is unclear, the run aborts with a clarification request.

## 4.4 GSEA-only Quick Start
```bash
python -m local_agent.cli \n  --gsea-csv GSEA_results.csv \n  --background-txt background_k562_example.txt \n  --output-dir outputs/gsea_only_demo \n  --gemini-api-key $(cat gemini_api_key.txt) \n  --gsea-only
```
Outputs are marked as hypotheses and include a reminder to run ESEA when enhancer data becomes available.

## 5. Outputs
Successful runs create the following under `--output-dir`:
- `draft_connections.json` – Gemini-generated theme↔cell-type hypotheses
- `claims.json` – atomic claims derived from the draft connections
- `verification.json` and `verification_table.txt` – numeric verification outcomes (marked as “Hypothesis” when `--gsea-only` is used)
- `revised_connections.json` – supported/partial connections after filtering
- `mini_thesis.txt` – final narrative plus an evidence box and, when enhancer data is missing, a reminder to run ESEA

## 6. Troubleshooting
- **API key errors**: ensure `GOOGLE_API_KEY` is exported or `--gemini-api-key` is provided. The CLI surfaces a clear message if the key is missing.
- **Empty responses or MAX_TOKENS**: increase `--gemini-max-output-tokens` (e.g., 8192) or reduce input size.
- **Credential hygiene**: never commit `gemini_api_key.txt`; add it to your personal `.gitignore` if necessary.

With the environment, key, and data in place, rerun the CLI whenever you have new perturbation results—no local model downloads required.
