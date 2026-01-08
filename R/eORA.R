#' Map SNPs (rsIDs and/or genomic loci) to GeneHancer IDs using a BED file path
#'
#' GeneHancer BED is assumed to be BED0 (0-based start, end-exclusive).
#' Internally converted to 1-based inclusive coordinates for GenomicRanges.
#'
#' @param x Character vector of rsIDs and/or locus strings, OR a data.frame with either:
#'   (1) chr + pos columns, or (2) chr + start + end columns (case-insensitive).
#' @param gh_bed File path to a GeneHancer BED file with no header.
#' @param ghid_col Integer column index in the BED containing GHID/GeneID. Default 4.
#' @param input_type One of "auto", "rsid", "loc_string", "dataframe".
#' @param loc_regex Regex to parse locus strings.
#' @param include_sex_chr If FALSE (default), keep autosomes only. If TRUE, include X/Y/MT.
#' @param chr_style "chr" (default) uses "chr1" style; "nochr" uses "1" style.
#' @param ensembl_version Optional integer Ensembl version to pin SNP mapping (unused in useMart mode).
#' @param grch Genome reference (kept for API compatibility; not used in useMart mode).
#' @param return One of "ghid", "mapping", "by_input".
#' @param unique_ghid If TRUE (default) and return="ghid", return unique GHIDs.
#' @param quiet Suppress informational messages.
#'
#' @return Depending on return:
#'   - "ghid": character vector of GHIDs
#'   - "mapping": data.frame(INPUT, GHID)
#'   - "by_input": named list input -> GHID vector
#'
#' @export
snps_to_ghid <- function(
    x,
    gh_bed,
    ghid_col = 4L,
    input_type = c("auto", "rsid", "loc_string", "dataframe"),
    loc_regex = "^(chr)?([0-9]+|X|Y|MT|M)[:_](\\d+)(?:-(\\d+))?$",
    include_sex_chr = FALSE,
    chr_style = c("chr", "nochr"),
    ensembl_version = NULL,
    grch = 38,
    return = c("ghid", "mapping", "by_input"),
    unique_ghid = TRUE,
    quiet = TRUE
) {
  input_type <- match.arg(input_type)
  chr_style  <- match.arg(chr_style)
  return     <- match.arg(return)

  if (missing(x)) stop("x is required.")
  if (missing(gh_bed) || !is.character(gh_bed) || length(gh_bed) != 1L) {
    stop("gh_bed must be a single file path (character length 1).")
  }
  if (!file.exists(gh_bed)) stop("gh_bed not found: ", gh_bed)

  if (!requireNamespace("GenomicRanges", quietly = TRUE)) stop("Package 'GenomicRanges' is required.")
  if (!requireNamespace("IRanges", quietly = TRUE)) stop("Package 'IRanges' is required.")

  normalize_chr <- function(chr) {
    chr <- as.character(chr)
    chr <- gsub("^chr", "", chr, ignore.case = TRUE)
    chr[chr %in% c("M")] <- "MT"
    if (chr_style == "chr") paste0("chr", chr) else chr
  }

  keep_chr <- function(chr_any) {
    chr_nochr <- gsub("^chr", "", as.character(chr_any), ignore.case = TRUE)
    chr_nochr[chr_nochr %in% c("M")] <- "MT"
    if (include_sex_chr) {
      grepl("^[0-9]+$|^X$|^Y$|^MT$", chr_nochr)
    } else {
      grepl("^[0-9]+$", chr_nochr)
    }
  }

  # ---- read GeneHancer BED (fixed format: no header, BED0) ----
  gh_tab <- utils::read.table(
    gh_bed,
    sep = "\t",
    header = FALSE,
    quote = "",
    comment.char = "",
    stringsAsFactors = FALSE,
    fill = TRUE
  )
  if (ncol(gh_tab) < 3) stop("GeneHancer BED must have >= 3 columns (chr, start, end).")
  if (ghid_col > ncol(gh_tab)) stop("ghid_col exceeds number of BED columns: ", ncol(gh_tab))

  gh_chr   <- gh_tab[[1]]
  gh_start <- suppressWarnings(as.integer(gh_tab[[2]]))
  gh_end   <- suppressWarnings(as.integer(gh_tab[[3]]))
  gh_ghid  <- as.character(gh_tab[[ghid_col]])

  if (anyNA(gh_start) || anyNA(gh_end)) stop("GeneHancer BED start/end must be integers (BED0).")

  # BED0 -> 1-based inclusive (IRanges)
  gh_start_1 <- gh_start + 1L
  gh_end_1   <- gh_end

  enh_gr <- GenomicRanges::GRanges(
    seqnames = normalize_chr(gh_chr),
    ranges   = IRanges::IRanges(start = gh_start_1, end = gh_end_1),
    GHID     = gh_ghid
  )

  # ---- locus parsing helpers ----
  parse_loc_strings <- function(v) {
    v <- as.character(v)
    m <- regexec(loc_regex, v, perl = TRUE)
    parts <- regmatches(v, m)
    ok <- lengths(parts) > 0

    if (!any(ok)) {
      return(data.frame(input=character(), chr=character(), start=integer(), end=integer(),
                        stringsAsFactors = FALSE))
    }

    out <- do.call(rbind, lapply(which(ok), function(i) {
      p <- parts[[i]]
      chr   <- p[3]
      start <- suppressWarnings(as.integer(p[4]))
      end   <- if (!is.na(p[5]) && nzchar(p[5])) suppressWarnings(as.integer(p[5])) else start
      data.frame(input=v[i], chr=chr, start=start, end=end, stringsAsFactors=FALSE)
    }))
    out
  }

  make_snp_gr_from_loci_df <- function(df, label_col = "input") {
    ok <- keep_chr(df$chr) & !is.na(df$start) & !is.na(df$end)
    df <- df[ok, , drop = FALSE]
    if (!nrow(df)) return(NULL)

    GenomicRanges::GRanges(
      seqnames = normalize_chr(df$chr),
      ranges   = IRanges::IRanges(start = df$start, end = df$end),
      INPUT    = df[[label_col]]
    )
  }

  # ---- split x into rsIDs vs loci ----
  rsids <- character(0)
  loc_df <- NULL

  if (input_type == "dataframe") {
    if (!is.data.frame(x)) stop("input_type='dataframe' requires x to be a data.frame.")
    cols <- tolower(colnames(x))

    if (all(c("chr","pos") %in% cols)) {
      chr <- x[[which(cols=="chr")[1]]]
      pos <- x[[which(cols=="pos")[1]]]
      loc_df <- data.frame(
        input = paste0(chr, ":", pos),
        chr   = chr,
        start = suppressWarnings(as.integer(pos)),
        end   = suppressWarnings(as.integer(pos)),
        stringsAsFactors = FALSE
      )
    } else if (all(c("chr","start","end") %in% cols)) {
      chr   <- x[[which(cols=="chr")[1]]]
      start <- x[[which(cols=="start")[1]]]
      end   <- x[[which(cols=="end")[1]]]
      loc_df <- data.frame(
        input = paste0(chr, ":", start, "-", end),
        chr   = chr,
        start = suppressWarnings(as.integer(start)),
        end   = suppressWarnings(as.integer(end)),
        stringsAsFactors = FALSE
      )
    } else {
      stop("Data frame must have chr+pos or chr+start+end columns (case-insensitive).")
    }
  } else {
    v <- unique(as.character(x))
    v <- v[!is.na(v) & v != ""]
    if (!length(v)) {
      if (return == "mapping") return(data.frame(INPUT=character(), GHID=character(), stringsAsFactors = FALSE))
      return(character(0))
    }

    is_rsid <- grepl("^rs\\d+$", v, ignore.case = TRUE)

    if (input_type == "rsid") {
      rsids <- v
    } else if (input_type == "loc_string") {
      loc_df <- parse_loc_strings(v)
    } else { # auto
      rsids <- v[is_rsid]
      others <- v[!is_rsid]
      if (length(others)) loc_df <- parse_loc_strings(others)
    }
  }

  snp_gr_list <- list()

  if (!is.null(loc_df) && nrow(loc_df)) {
    gr_loc <- make_snp_gr_from_loci_df(loc_df, label_col = "input")
    if (!is.null(gr_loc)) snp_gr_list[["loc"]] <- gr_loc
  }

  # ---- rsID -> chr/pos via biomaRt (stable useMart mode) ----
  # biomaRt SNP coordinates are expected to be 1-based; if zeros appear, shift +1 defensively.
  if (length(rsids)) {
    if (!requireNamespace("biomaRt", quietly = TRUE)) {
      stop("biomaRt is required to map rsIDs; install it or provide loci instead of rsIDs.")
    }

    snp_mart <- biomaRt::useMart("ENSEMBL_MART_SNP", dataset = "hsapiens_snp")

    loc <- biomaRt::getBM(
      attributes = c("refsnp_id","chr_name","chrom_start"),
      filters    = "snp_filter",
      values     = rsids,
      mart       = snp_mart
    )

    if (nrow(loc)) {
      loc$chrom_start <- suppressWarnings(as.integer(loc$chrom_start))
      if (any(loc$chrom_start == 0L, na.rm = TRUE)) {
        if (!quiet) message("biomaRt chrom_start appears 0-based; shifting to 1-based coordinates.")
        loc$chrom_start <- loc$chrom_start + 1L
      }
      ok <- !is.na(loc$chrom_start) & keep_chr(loc$chr_name)
      loc <- loc[ok, , drop = FALSE]

      if (nrow(loc)) {
        snp_gr_list[["rsid"]] <- GenomicRanges::GRanges(
          seqnames = normalize_chr(loc$chr_name),
          ranges   = IRanges::IRanges(start = loc$chrom_start, width = 1),
          INPUT    = loc$refsnp_id
        )
      } else if (!quiet) {
        message("All rsID mappings filtered out by chromosome settings.")
      }
    } else if (!quiet) {
      message("No rsIDs mapped by biomaRt.")
    }
  }

  if (!length(snp_gr_list)) {
    if (!quiet) message("No usable SNP inputs after parsing/mapping.")
    if (return == "mapping") return(data.frame(INPUT=character(), GHID=character(), stringsAsFactors = FALSE))
    return(character(0))
  }

  # SAFE: ensure snp_gr is a GRanges, never a list
  snp_gr <- snp_gr_list[[1]]
  if (length(snp_gr_list) > 1) {
    for (i in 2:length(snp_gr_list)) {
      snp_gr <- c(snp_gr, snp_gr_list[[i]])
    }
  }

  hits <- GenomicRanges::findOverlaps(snp_gr, enh_gr, ignore.strand = TRUE)

  if (!length(hits)) {
    if (!quiet) message("No overlaps with GeneHancer intervals.")
    if (return == "mapping") return(data.frame(INPUT=character(), GHID=character(), stringsAsFactors = FALSE))
    return(character(0))
  }

  mapping <- unique(data.frame(
    INPUT = snp_gr$INPUT[S4Vectors::queryHits(hits)],
    GHID  = enh_gr$GHID[S4Vectors::subjectHits(hits)],
    stringsAsFactors = FALSE
  ))

  if (return == "mapping") return(mapping)
  if (return == "by_input") return(split(mapping$GHID, mapping$INPUT))

  out <- mapping$GHID
  if (unique_ghid) out <- unique(out)
  out
}


#' Run enhancer overlap enrichment analysis (eORA) from SNPs to enhancer sets
#'
#' @param snps SNP inputs (same formats as snps_to_ghid x).
#' @param gh_bed File path to GeneHancer BED (no header; BED0).
#' @param enhancer_sets Named list: set_name -> GHID vector.
#' @param ghid_col BED column for GHID. Default 4.
#' @param B Permutations. Default 10000.
#' @param seed Random seed.
#' @param alpha Significance threshold. Default 0.05.
#' @param p_adjust_method p.adjust method for empirical p-values; default "BH". Use NULL to skip.
#' @param ... Passed to snps_to_ghid (e.g., input_type, include_sex_chr, chr_style, etc.).
#'
#' @return data.frame with one row per enhancer set.
#' @export
run_eORA <- function(
    snps,
    gh_bed,
    enhancer_sets,
    ghid_col = 4L,
    B = 10000L,
    seed = 123L,
    alpha = 0.05,
    p_adjust_method = "BH",
    ...
) {
  if (missing(snps)) stop("snps is required.")
  if (missing(gh_bed) || !is.character(gh_bed) || length(gh_bed) != 1L) {
    stop("gh_bed must be a single file path (character length 1).")
  }
  if (!file.exists(gh_bed)) stop("gh_bed not found: ", gh_bed)

  if (is.null(enhancer_sets) || !is.list(enhancer_sets) || is.null(names(enhancer_sets))) {
    stop("enhancer_sets must be a named list (names are set names; values are GHID vectors).")
  }

  enhancer_sets <- lapply(enhancer_sets, function(v) {
    v <- trimws(as.character(v))
    v <- v[!is.na(v) & v != ""]
    unique(v)
  })
  enhancer_sets <- enhancer_sets[lengths(enhancer_sets) > 0]
  set_names <- names(enhancer_sets)
  if (!length(set_names)) stop("enhancer_sets is empty after cleaning.")

  U <- sort(unique(unlist(enhancer_sets, use.names = FALSE)))
  U <- U[!is.na(U) & U != ""]
  N <- length(U)
  if (N == 0) stop("Universe U is empty.")

  S_all <- snps_to_ghid(
    x = snps,
    gh_bed = gh_bed,
    ghid_col = ghid_col,
    return = "ghid",
    unique_ghid = TRUE,
    ...
  )
  S <- intersect(S_all, U)
  K <- length(S)

  M_vec <- lengths(enhancer_sets)
  x_obs <- vapply(enhancer_sets, function(es) length(intersect(es, S)), integer(1))

  if (K == 0) {
    return(data.frame(
      enhancer_set = set_names,
      N = rep.int(N, length(set_names)),
      M = as.integer(M_vec),
      K = rep.int(0L, length(set_names)),
      x = as.integer(x_obs),
      p_hyper = rep.int(NA_real_, length(set_names)),
      p_empirical = rep.int(NA_real_, length(set_names)),
      p_empirical_adj = rep.int(NA_real_, length(set_names)),
      significant = rep.int(FALSE, length(set_names)),
      stringsAsFactors = FALSE
    ))
  }

  p_hyper <- stats::phyper(
    q = pmax(0L, x_obs) - 1L,
    m = M_vec,
    n = N - M_vec,
    k = K,
    lower.tail = FALSE
  )

  set.seed(seed)
  ge_count <- integer(length(set_names))

  for (b in seq_len(B)) {
    S_b <- sample(U, size = K, replace = FALSE)
    x_b <- vapply(enhancer_sets, function(es) length(intersect(es, S_b)), integer(1))
    ge_count <- ge_count + as.integer(x_b >= x_obs)
  }

  p_empirical <- (ge_count + 1) / (B + 1)

  p_empirical_adj <- if (is.null(p_adjust_method)) {
    rep.int(NA_real_, length(set_names))
  } else {
    stats::p.adjust(p_empirical, method = p_adjust_method)
  }

  significant <- if (is.null(p_adjust_method)) {
    p_empirical <= alpha
  } else {
    p_empirical_adj <= alpha
  }

  out <- data.frame(
    enhancer_set = set_names,
    N = rep.int(N, length(set_names)),
    M = as.integer(M_vec),
    K = rep.int(as.integer(K), length(set_names)),
    x = as.integer(x_obs),
    p_hyper = as.numeric(p_hyper),
    p_empirical = as.numeric(p_empirical),
    p_empirical_adj = as.numeric(p_empirical_adj),
    significant = as.logical(significant),
    stringsAsFactors = FALSE
  )

  key <- if (is.null(p_adjust_method)) out$p_empirical else out$p_empirical_adj
  out[order(key, out$p_empirical, out$p_hyper, -out$x, na.last = TRUE), , drop = FALSE]
}
