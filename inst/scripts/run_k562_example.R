#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  if (requireNamespace('pkgload', quietly = TRUE)) {
    pkgload::load_all('.')
  } else {
    library(ENSEMBLE)
  }
})

message('Starting example workflow...')
args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) && nzchar(args[1])) {
  path.expand(args[1])
} else {
  file.path(getwd(), 'outputs', 'example_run')
}
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}
message('Saving results to: ', output_dir)

# ------------------------------------------------------------------
# File locations (override via example_config.json if needed)
# ------------------------------------------------------------------
example_dir <- ensemble_example_data()
if (!nzchar(example_dir)) {
  stop('Example data not found. Install ENSEMBLE or provide a config JSON.')
}

default_config <- list(
  counts_file_enh = file.path(example_dir, "example_enhancer_counts.tsv"),
  counts_file_gene = file.path(example_dir, "example_gene_counts.tsv"),
  genehancer_file = file.path(example_dir, "example_genehancer_annotations.tsv"),
  gencode_file = file.path(example_dir, "example_gene_annotations.gtf"),
  enhancer_set_file = file.path(example_dir, "example_enhancer_sets.gmt"),
  metadata_file = file.path(example_dir, "example_metadata.csv"),
  msigdb_files = c(
    file.path(example_dir, "example_msigdb_hallmarks.gmt"),
    file.path(example_dir, "example_msigdb_custom.gmt")
  ),
  contrast = c("group", "SNAI1_KO", "WT"),
  target_groups = NULL,
  group_column = "group",
  source_column = "source_name"
)

config_file <- file.path(getwd(), "example_config.json")
cfg <- default_config
if (file.exists(config_file)) {
  message('Loading overrides from ', config_file)
  override <- jsonlite::fromJSON(config_file, simplifyVector = TRUE)
  cfg <- utils::modifyList(cfg, override)
}

counts_file_enh <- cfg$counts_file_enh
counts_file_gene <- cfg$counts_file_gene
genehancer_file <- cfg$genehancer_file
gencode_file <- cfg$gencode_file
enhancer_set_file <- cfg$enhancer_set_file
metadata_file <- cfg$metadata_file
msigdb_files <- cfg$msigdb_files
contrast <- cfg$contrast
target_groups <- cfg$target_groups
group_col <- if (!is.null(cfg$group_column)) cfg$group_column else "group"
source_col <- if (!is.null(cfg$source_column)) cfg$source_column else "source_name"

required_paths <- c(
  counts_file_enh,
  counts_file_gene,
  genehancer_file,
  gencode_file,
  enhancer_set_file,
  metadata_file,
  msigdb_files
)
missing <- required_paths[!file.exists(required_paths)]
if (length(missing)) {
  stop('Missing required input files:\n', paste(missing, collapse = '\n'))
}

# ------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------
meta_data <- utils::read.csv(metadata_file, row.names = 1, stringsAsFactors = FALSE)
if (!source_col %in% colnames(meta_data)) {
  stop('Column `', source_col, '` not found in metadata file: ', metadata_file)
}
if (length(target_groups)) {
  meta_data <- meta_data[meta_data[[source_col]] %in% target_groups, , drop = FALSE]
}
if (!nrow(meta_data)) {
  stop('No samples matched the requested metadata filters.')
}
if (!group_col %in% colnames(meta_data)) {
  meta_data[[group_col]] <- gsub(" ", "_", meta_data[[source_col]], fixed = TRUE)
}

# ------------------------------------------------------------------
# Enhancer counts preprocessing
# ------------------------------------------------------------------
counts_dt <- data.table::fread(counts_file_enh, check.names = FALSE)
counts_matrix <- as.matrix(data.frame(counts_dt, row.names = 1, check.names = FALSE))
samples_keep <- colnames(counts_matrix)[colnames(counts_matrix) %in% rownames(meta_data)]
if (!length(samples_keep)) {
  stop('None of the targeted samples were found in the enhancer counts matrix.')
}
meta_missing <- setdiff(rownames(meta_data), samples_keep)
if (length(meta_missing)) {
  stop('Enhancer counts matrix is missing expected samples: ', paste(meta_missing, collapse = ', '))
}
counts_matrix <- counts_matrix[, samples_keep, drop = FALSE]
meta_data <- meta_data[samples_keep, , drop = FALSE]

group <- meta_data[[group_col]]

prep <- preprocess_enhancerMatrix(
  enhancer.counts = counts_matrix,
  filter.low = TRUE,
  GeneHancer.file = genehancer_file,
  GeneCode.file = gencode_file,
  meta_data = meta_data,
  group.col = group_col,
  CPM_tMM = TRUE
)
if (is.list(prep)) {
  counts_matrix_filtered <- prep$counts
  cpm_tmm <- prep$cpm_tmm
} else {
  counts_matrix_filtered <- prep
  cpm_tmm <- NULL
}
if (!nrow(counts_matrix_filtered)) {
  stop('No enhancers remained after preprocessing; adjust filtering parameters if needed.')
}

# ------------------------------------------------------------------
# Metadata export (aligned order)
# ------------------------------------------------------------------
metadata_out <- data.table::data.table(
  Sample = rownames(meta_data),
  source_name = meta_data[[source_col]],
  group = group
)
data.table::fwrite(metadata_out, file.path(output_dir, "metadata.csv"))

# ------------------------------------------------------------------
# Differential enhancer analysis
# ------------------------------------------------------------------

DEE <- ENSEMBLE.DEE(
  eRNA.counts = prep$counts,
  meta_data = meta_data,
  contrast = contrast,
  round.counts = TRUE,
  min.countsum = 10
)
weight_enh <- DE.weights(DEE)

# ------------------------------------------------------------------
# Enhancer set filtering and ESEA
# ------------------------------------------------------------------
enhancer_sets <- read_concepts(enhancer_set_file)


enhancer_sets_filtered <- retain_specific_enhancers(enhancer_sets, maxMulti = 0.1, minSize = 50)

ESEA_results <- ESEA_fast(weight_enh, compare.list = enhancer_sets_filtered)
ESEA_results <- data.table::data.table(ESEA_results)
if (nrow(ESEA_results)) {
  ESEA_results[, leadingEdge := vapply(
    leadingEdge,
    function(x) if (length(x)) paste(x, collapse = ", ") else NA_character_,
    character(1)
  )]
}
data.table::fwrite(ESEA_results, file.path(output_dir, "ESEA_results.csv"))

# ------------------------------------------------------------------
# Gene-level GSEA
# ------------------------------------------------------------------
counts_dt_gene <- data.table::fread(counts_file_gene, check.names = FALSE)
counts_matrix_gene <- as.matrix(data.frame(counts_dt_gene, row.names = 1, check.names = FALSE))
missing_gene <- setdiff(rownames(meta_data), colnames(counts_matrix_gene))
if (length(missing_gene)) {
  stop('Gene expression matrix is missing expected samples: ', paste(missing_gene, collapse = ', '))
}
counts_matrix_gene <- counts_matrix_gene[, rownames(meta_data), drop = FALSE]

y_gene <- edgeR::DGEList(counts = counts_matrix_gene, group = group)
keep_gene <- edgeR::filterByExpr(y_gene, group = group)
if (!any(keep_gene)) {
  stop('No genes passed filterByExpr for GSEA input.')
}
y_gene <- y_gene[keep_gene, , keep.lib.sizes = FALSE]

gene_sets <- lapply(msigdb_files, read_concepts)
gene_sets <- do.call(c, gene_sets)
if (!length(gene_sets)) {
  stop('No gene sets were loaded from msigdb_files.')
}

DGE <- ENSEMBLE.DEE(
  eRNA.counts = y_gene$counts,
  meta_data = meta_data,
  contrast = contrast,
  round.counts = TRUE,
  min.countsum = 10
)
weight_gene <- DE.weights(DGE)

GSEA_results <- ESEA_fast(weight_gene, compare.list = gene_sets)
GSEA_results <- data.table::data.table(GSEA_results)
if (nrow(GSEA_results)) {
  GSEA_results[, leadingEdge := vapply(
    leadingEdge,
    function(x) if (length(x)) paste(x, collapse = ", ") else NA_character_,
    character(1)
  )]
}
data.table::fwrite(GSEA_results, file.path(output_dir, "GSEA_results.csv"))

message('Example workflow completed successfully.')
