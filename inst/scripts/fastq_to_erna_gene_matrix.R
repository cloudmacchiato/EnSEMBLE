#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (!requireNamespace("optparse", quietly = TRUE)) {
    stop("Package `optparse` is required. Install it with install.packages('optparse').")
  }
  library(optparse)
  library(Rsubread)
  library(edgeR)
  library(data.table)
  library(tools)
})

option_list <- list(
  make_option(c("-f", "--fastq-dir"), type = "character", help = "Directory containing FASTQ files organised as <fastq-dir>/<sample>/*.fastq(.gz)"),
  make_option(c("-o", "--output-dir"), type = "character", default = "expression", help = "Directory to store per-sample outputs and merged matrices [default: %default]"),
  make_option(c("-p", "--prefix"), type = "character", default = "project", help = "Prefix used when writing merged expression matrices [default: %default]"),
  make_option(c("-l", "--library-type"), type = "character", default = "total", help = "Library type: 'total' for stranded total RNA-seq or 'mrna' for mRNA-seq [default: %default]"),
  make_option(c("-s", "--strandness"), type = "integer", default = NA_integer_, help = "Override strandedness (0=unstranded, 1=forward, 2=reverse). By default inferred from library type."),
  make_option(c("-r", "--reference-fasta"), type = "character",
              default = "/ix/xiaosongwang/wangx13/Pipeline/built-in-files/Rsubread/GRCh38.primary_assembly.genome.fa",
              help = "Reference genome FASTA used for Rsubread alignment [default: %default]"),
  make_option(c("-g", "--gene-gtf"), type = "character",
              default = "/ix/xiaosongwang/wangx13/Pipeline/built-in-files/Rsubread/gencode.v47.basic.annotation.gtf",
              help = "Gene annotation GTF (Gencode v47 basic recommended) [default: %default]"),
  make_option(c("-e", "--erna-saf"), type = "character",
              default = "/ix/xiaosongwang/wangx13/Pipeline/built-in-files/Rsubread/GeneHancer_v5.20.hg38.saf",
              help = "GeneHancer enhancer SAF file for eRNA quantification [default: %default]"),
  make_option(c("-t", "--threads"), type = "integer", default = 4L, help = "Threads passed to Rsubread align/featureCounts [default: %default]"),
  make_option(c("--keep-bam"), action = "store_true", default = FALSE, help = "Retain aligned BAM files after counting [default: remove]"),
  make_option(c("--skip-existing"), action = "store_true", default = FALSE, help = "Skip samples that already contain featureCounts output files [default: %default]")
)

opts <- parse_args(OptionParser(option_list = option_list))

stopifnot(!is.null(opts$`fastq-dir`))
fastq_dir <- normalizePath(opts$`fastq-dir`, mustWork = TRUE)
output_dir <- normalizePath(opts$`output-dir`, mustWork = FALSE)
prefix <- opts$prefix
library_type <- tolower(opts$`library-type`)
strandness <- opts$strandness

if (is.na(strandness)) {
  strandness <- switch(
    library_type,
    total = 2L,      # Illumina TruSeq stranded total RNA (reverse)
    mrna = 2L,       # Illumina TruSeq stranded mRNA (reverse)
    stop("Unknown library type '", opts$`library-type`, "'. Use 'total' or 'mrna', or set --strandness explicitly.")
  )
} else if (!strandness %in% c(0L, 1L, 2L)) {
  stop("--strandness must be one of 0, 1, or 2.")
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

ensure_reference_index <- function(reference_fasta) {
  index_prefix <- reference_fasta
  index_reads <- paste0(index_prefix, ".reads")
  if (!file.exists(index_reads)) {
    message("Reference index not found at ", index_reads, ". Building Rsubread index (one-time cost)...")
    buildindex(basename = index_prefix, reference = reference_fasta, memory = 64000000000)
  }
  index_prefix
}

detect_samples <- function(fastq_root) {
  sample_dirs <- list.dirs(fastq_root, recursive = FALSE, full.names = TRUE)
  sample_dirs <- sample_dirs[basename(sample_dirs) != "logs"]
  if (!length(sample_dirs)) {
    stop("No sample directories found under ", fastq_root, ". Expect layout <fastq-dir>/<sample>/*.fastq(.gz)")
  }
  sample_list <- lapply(sample_dirs, function(sample_path) {
    sample_name <- basename(sample_path)
    fastqs <- list.files(sample_path, pattern = "\\.(fastq|fq)(\\.gz)?$", full.names = TRUE)
    if (!length(fastqs)) {
      stop("No FASTQ files detected in ", sample_path)
    }
    r1 <- fastqs[grepl("(_R?1\\b|_1\\b)", fastqs, ignore.case = TRUE)]
    r2 <- fastqs[grepl("(_R?2\\b|_2\\b)", fastqs, ignore.case = TRUE)]
    if (length(r1) == 0 && length(r2) == 0) {
      fastqs <- sort(fastqs)
      if (length(fastqs) == 1) {
        r1 <- fastqs
        r2 <- character(0)
      } else if (length(fastqs) == 2) {
        r1 <- fastqs[1]
        r2 <- fastqs[2]
      } else {
        stop("Unable to infer read pairing for sample ", sample_name, ". Provide files with _R1/_R2 or _1/_2 suffixes.")
      }
    }
    if (length(r2) > 1 && length(r1) == length(r2)) {
      # Keep naming consistency for multi-lane data; assume matching order
      r1 <- sort(r1)
      r2 <- sort(r2)
    }
    if (length(r2) && length(r1) != length(r2)) {
      stop("Mismatched number of R1 and R2 files for sample ", sample_name)
    }
    list(
      sample = sample_name,
      r1 = sort(r1),
      r2 = sort(r2)
    )
  })
  names(sample_list) <- vapply(sample_list, `[[`, character(1), "sample")
  sample_list
}

run_alignment_if_needed <- function(sample_info, output_sample_dir, reference_index, threads) {
  bam_path <- file.path(output_sample_dir, paste0(sample_info$sample, ".bam"))
  if (file.exists(bam_path)) {
    message("Skipping alignment for ", sample_info$sample, "; BAM already present.")
    return(bam_path)
  }
  input_format <- if (all(grepl("\\.bam$", c(sample_info$r1, sample_info$r2)))) "BAM" else if (all(grepl("\\.gz$", sample_info$r1))) "gzFASTQ" else "FASTQ"
  paired <- length(sample_info$r2) > 0
  message("Aligning sample ", sample_info$sample, if (paired) " (paired-end)" else " (single-end)")
  align(
    index = reference_index,
    readfile1 = sample_info$r1,
    readfile2 = if (paired) sample_info$r2 else NULL,
    input_format = input_format,
    output_file = bam_path,
    nthreads = threads,
    unique = TRUE,
    indels = 2
  )
  bam_path
}

write_metrics <- function(sample_name, output_dir, suffix, counts_list) {
  out_path <- file.path(output_dir, paste0(sample_name, ".", suffix))
  data_to_write <- data.frame(Feature = rownames(counts_list), counts_list, stringsAsFactors = FALSE, check.names = FALSE)
  fwrite(data_to_write, file = out_path, quote = FALSE, sep = "\t", col.names = FALSE)
  message("  wrote ", basename(out_path))
}

quantify_gene_expression <- function(bam_path, sample_info, output_sample_dir, gtf_path, threads, strandness, paired_end) {
  attr_map <- c(gene_name = "HGNC", gene_id = "ENSG", exon_id = "ENSE")
  process_attr_type <- function(attr_type) {
    featureCounts(
      files = bam_path,
      annot.ext = gtf_path,
      isGTFAnnotationFile = TRUE,
      GTF.attrType = attr_type,
      strandSpecific = strandness,
      isPairedEnd = paired_end,
      nthreads = threads
    )
  }
  lapply(names(attr_map), function(attr_type) {
    fc <- process_attr_type(attr_type)
    counts <- fc$counts
    lengths <- fc$annotation$Length
    counts_len_norm <- counts / lengths
    tpm <- sweep(counts_len_norm, 2, colSums(counts_len_norm), "/") * 1e6
    dge <- DGEList(counts = counts, genes = fc$annotation)
    rpkm <- rpkm(dge, dge$genes$Length)
    cpm <- cpm(dge)
    stats_table <- data.frame(fc$stat[-1], row.names = fc$stat$Status)
    feature_label <- attr_map[[attr_type]]
    write_metrics(sample_info$sample, output_sample_dir, paste0(feature_label, ".featureCounts"), counts)
    write_metrics(sample_info$sample, output_sample_dir, paste0(feature_label, ".unlog.tpm"), tpm)
    write_metrics(sample_info$sample, output_sample_dir, paste0(feature_label, ".unlog.rpkm"), rpkm)
    write_metrics(sample_info$sample, output_sample_dir, paste0(feature_label, ".unlog.cpm"), cpm)
    stats_out <- file.path(output_sample_dir, paste0(sample_info$sample, ".", feature_label, ".stats"))
    fwrite(data.frame(Status = rownames(stats_table), stats_table, stringsAsFactors = FALSE), file = stats_out, quote = FALSE, sep = "\t", col.names = TRUE)
    message("Completed gene quantification for ", feature_label)
    invisible(NULL)
  })
}

quantify_enhancer_expression <- function(bam_path, sample_info, output_sample_dir, saf_path, threads, strandness, paired_end) {
  fc <- featureCounts(
    files = bam_path,
    annot.ext = saf_path,
    isGTFAnnotationFile = FALSE,
    strandSpecific = strandness,
    isPairedEnd = paired_end,
    nthreads = threads
  )
  counts <- fc$counts
  lengths <- fc$annotation$Length
  counts_len_norm <- counts / lengths
  tpm <- sweep(counts_len_norm, 2, colSums(counts_len_norm), "/") * 1e6
  dge <- DGEList(counts = counts, genes = fc$annotation)
  rpkm <- rpkm(dge, dge$genes$Length)
  cpm <- cpm(dge)
  stats_table <- data.frame(fc$stat[-1], row.names = fc$stat$Status)
  write_metrics(sample_info$sample, output_sample_dir, "GeneHancer_eRNA.featureCounts", counts)
  write_metrics(sample_info$sample, output_sample_dir, "GeneHancer_eRNA.unlog.tpm", tpm)
  write_metrics(sample_info$sample, output_sample_dir, "GeneHancer_eRNA.unlog.rpkm", rpkm)
  write_metrics(sample_info$sample, output_sample_dir, "GeneHancer_eRNA.unlog.cpm", cpm)
  stats_out <- file.path(output_sample_dir, paste0(sample_info$sample, ".GeneHancer_eRNA.stats"))
  fwrite(data.frame(Status = rownames(stats_table), stats_table, stringsAsFactors = FALSE), file = stats_out, quote = FALSE, sep = "\t", col.names = TRUE)
  message("Completed enhancer quantification for GeneHancer_eRNA")
  invisible(NULL)
}

sample_complete <- function(sample_name, output_sample_dir) {
  required_files <- c(
    file.path(output_sample_dir, paste0(sample_name, ".HGNC.featureCounts")),
    file.path(output_sample_dir, paste0(sample_name, ".GeneHancer_eRNA.featureCounts"))
  )
  all(file.exists(required_files))
}

merge_expression_tables <- function(output_root, features, suffixes, prefix) {
  sample_dirs <- list.dirs(output_root, recursive = FALSE, full.names = TRUE)
  sample_dirs <- sample_dirs[basename(sample_dirs) != "logs"]
  if (!length(sample_dirs)) {
    warning("No per-sample directories found under ", output_root, "; skipping merge.")
    return(invisible(NULL))
  }
  sample_names <- basename(sample_dirs)
  for (feature in features) {
    for (suffix in suffixes) {
      file_paths <- file.path(sample_dirs, paste0(sample_names, ".", feature, ".", suffix))
      existing <- file_paths[file.exists(file_paths)]
      if (!length(existing)) {
        warning("Skipping merge for ", feature, " ", suffix, "; no input files detected.")
        next
      }
      tables <- lapply(existing, function(x) fread(x, header = FALSE))
      first_table <- tables[[1]]
      matrix_data <- sapply(tables, function(tbl) as.numeric(tbl[[2]]))
      if (is.null(dim(matrix_data))) {
        matrix_data <- matrix(matrix_data, ncol = 1)
      }
      rownames(matrix_data) <- first_table[[1]]
      colnames(matrix_data) <- basename(dirname(existing))
      nonzero <- rowSums(matrix_data) != 0
      matrix_data <- matrix_data[nonzero, , drop = FALSE]
      merged <- data.table(Name = rownames(matrix_data))
      merged <- cbind(merged, as.data.table(matrix_data))
      setnames(merged, old = names(merged)[-1], new = colnames(matrix_data))
      out_file <- file.path(output_root, paste0(prefix, "_", feature, "-expr-", suffix, ".tsv"))
      fwrite(merged, out_file, sep = "\t", quote = FALSE)
      message("Merged matrix written: ", out_file)
    }
  }
}

reference_index <- ensure_reference_index(opts$`reference-fasta`)
samples <- detect_samples(fastq_dir)

for (sample_info in samples) {
  sample_out_dir <- file.path(output_dir, sample_info$sample)
  dir.create(sample_out_dir, recursive = TRUE, showWarnings = FALSE)
  if (opts$`skip-existing` && sample_complete(sample_info$sample, sample_out_dir)) {
    message("Skipping quantification for ", sample_info$sample, " (existing outputs detected).")
    next
  }
  bam_path <- run_alignment_if_needed(sample_info, sample_out_dir, reference_index, opts$threads)
  paired_end <- length(sample_info$r2) > 0
  quantify_gene_expression(bam_path, sample_info, sample_out_dir, opts$`gene-gtf`, opts$threads, strandness, paired_end)
  quantify_enhancer_expression(bam_path, sample_info, sample_out_dir, opts$`erna-saf`, opts$threads, strandness, paired_end)
  if (!opts$`keep-bam` && file.exists(bam_path)) {
    file.remove(bam_path)
  }
}

merge_expression_tables(
  output_root = output_dir,
  features = c("HGNC", "GeneHancer_eRNA"),
  suffixes = c("featureCounts", "unlog.tpm", "unlog.cpm", "unlog.rpkm"),
  prefix = paste(prefix, library_type, sep = "_")
)

message("FASTQ-to-matrix workflow complete.")
