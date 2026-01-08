#' Read Concepts from a File
#'
#' Reads a file and processes genotype data, filtering out entries with fewer than a specified minimum.
#'
#' @param fileName A character string specifying the file path.
#' @param min An integer specifying the minimum number of genotype elements (default is 5).
#'
#' @return A named list where each element contains genotype information.
#' @importFrom data.table fread
#' @export
read_concepts <- function(fileName, min = 5){
  con <- file(fileName, open = "r", encoding = "latin1")
  on.exit(close(con), add = TRUE)

  genotype.list <- list()
  genotype.names <- character()
  idx <- 0L
  first_line <- TRUE

  repeat {
    line <- readLines(con, n = 1L, warn = FALSE)
    if (!length(line)) break

    if (first_line && grepl("^\\s*#", line, useBytes = TRUE)) {
      first_line <- FALSE
      next
    }
    first_line <- FALSE

    line <- iconv(line, from = "latin1", to = "ASCII//TRANSLIT")
    if (is.na(line) || !nzchar(line)) next

    genotype <- strsplit(line, "\t", fixed = TRUE)[[1]]
    if (length(genotype) < 3L) next

    gname <- genotype[1L]
    members <- genotype[-c(1L, 2L)]
    if (length(members) < min) next

    idx <- idx + 1L
    genotype.list[[idx]] <- members
    genotype.names[idx] <- gname
  }

  if (length(genotype.list)) {
    names(genotype.list) <- genotype.names
  }
  genotype.list
}

#' Retain Enhancer Sets Enriched for Specific Loci
#'
#' Filters enhancer sets to keep loci that appear in relatively few sets and
#' removes sets that become too small after filtering.
#'
#' @param enhancer_sets Named list of enhancer sets where each element is a
#'   character vector of enhancer IDs.
#' @param maxMulti Maximum fraction of sets an enhancer can belong to before it
#'   is discarded. Default is \code{0.05}.
#' @param minSize Minimum number of enhancers required to keep a set after
#'   filtering. Default is \code{10}.
#'
#' @return Named list containing only the retained enhancer sets.
#' @importFrom data.table data.table rbindlist
#' @export
retain_specific_enhancers <- function(enhancer_sets, maxMulti = 0.05, minSize = 10) {
  if (!length(enhancer_sets)) {
    return(enhancer_sets)
  }

  enhancer_set_map <- unique(data.table::rbindlist(lapply(names(enhancer_sets), function(tf) {
    data.table::data.table(enhancer = enhancer_sets[[tf]], tf = tf)
  })))

  if (!nrow(enhancer_set_map)) {
    return(enhancer_sets[0])
  }

  enhancer_map_count <- enhancer_set_map[, .N, by = enhancer]
  total_sets <- length(unique(enhancer_set_map$tf))
  enhancer_map_count[, frequency := N / total_sets]

  valid_enhancers <- enhancer_map_count[frequency <= maxMulti, enhancer]
  filtered_sets <- lapply(enhancer_sets, function(enhancers) enhancers[enhancers %in% valid_enhancers])
  filtered_sets <- filtered_sets[lengths(filtered_sets) > minSize]
  filtered_sets
}

#' Map Locus-Indexed Enhancer Expression to GeneHancer IDs (Sparse-friendly)
#'
#' @param enhancer.exp Numeric matrix/data.frame or a Matrix::dgCMatrix/dgTMatrix.
#'   Rownames must be "chr:start-end".
#' @param GH.saf data.frame with columns Chr, Start, End, GeneID.
#' @param collapse.rows logical; if TRUE, aggregate loci mapping to the same GeneID.
#' @param agg One of c("trimmed_mean","mean","sum"). Default: "trimmed_mean"
#'        (for compatibility); for huge matrices prefer "mean" or "sum".
#' @param trim Proportion to trim from each tail when agg = "trimmed_mean". Default 0.1.
#' @param return_sparse If TRUE and possible, return a sparse Matrix (dgCMatrix).
#'        For agg %in% c("mean","sum"), sparse is always used.
#' @return When input is sparse and collapse.rows=TRUE, returns a sparse dgCMatrix by default.
#'         Otherwise returns a base matrix/data.frame analogous to the original.
#' @importFrom data.table data.table tstrsplit
#' @importFrom GenomicRanges GRanges findOverlaps mcols seqnames start end
#' @importFrom IRanges IRanges
#' @importFrom S4Vectors queryHits subjectHits
#' @import Matrix
#' @export exp_loci2Ghid_sparse
#' @export exp.loci2Ghid_sparse
exp_loci2Ghid_sparse <- function(enhancer.exp, GH.saf,
                                 collapse.rows = TRUE,
                                 agg = c("trimmed_mean","mean","sum"),
                                 trim = 0.1,
                                 return_sparse = TRUE) {
  agg <- match.arg(agg)

  # --- checks
  rn <- rownames(enhancer.exp)
  if (is.null(rn) || anyNA(rn)) {
    stop("`enhancer.exp` must have non-NA rownames in the form 'chr:start-end'.")
  }
  req_cols <- c("Chr","Start","End","GeneID")
  miss <- setdiff(req_cols, colnames(GH.saf))
  if (length(miss)) stop("`GH.saf` missing: ", paste(miss, collapse = ", "))

  is_sparse <- inherits(enhancer.exp, "dgCMatrix") || inherits(enhancer.exp, "dgTMatrix")
  if (!is_sparse) {
    if (is.data.frame(enhancer.exp)) {
      # keep numeric only, avoid factors
      enhancer.exp <- as.matrix(data.frame(enhancer.exp, check.names = FALSE))
    } else if (!is.matrix(enhancer.exp)) {
      stop("`enhancer.exp` must be a matrix/data.frame or a Matrix::dg*CMatrix.")
    }
  } else {
    if (inherits(enhancer.exp, "dgTMatrix")) {
      enhancer.exp <- as(enhancer.exp, "dgCMatrix") # use CSC layout
    }
  }

  # --- build GRanges for loci
  loci.dt <- data.table::data.table(coord = rn)
  loci.dt[, c("seqnames","start","end") := data.table::tstrsplit(coord, "[\\:\\-]", type.convert = TRUE)]
  if (anyNA(loci.dt$seqnames) || anyNA(loci.dt$start) || anyNA(loci.dt$end)) {
    bad <- loci.dt$coord[is.na(loci.dt$seqnames) | is.na(loci.dt$start) | is.na(loci.dt$end)]
    stop("Malformed rownames. Examples: ", paste(utils::head(bad, 3), collapse = ", "))
  }

  enh.gr <- GenomicRanges::GRanges(
    seqnames = loci.dt$seqnames,
    ranges   = IRanges::IRanges(start = loci.dt$start, end = loci.dt$end),
    strand   = "*"
  )

  gh.gr <- GenomicRanges::GRanges(
    seqnames = GH.saf$Chr,
    ranges   = IRanges::IRanges(start = as.integer(GH.saf$Start), end = as.integer(GH.saf$End)),
    strand   = "*"
  )
  S4Vectors::mcols(gh.gr)$GeneID <- GH.saf$GeneID

  # --- overlaps: map each locus (row) to the first GH GeneID
  hits <- GenomicRanges::findOverlaps(enh.gr, gh.gr)
  if (length(hits) == 0L) stop("No overlaps between enhancer loci and GH.saf.")

  qh <- S4Vectors::queryHits(hits)
  sh <- S4Vectors::subjectHits(hits)
  keep_first <- !duplicated(qh)
  qh <- qh[keep_first]; sh <- sh[keep_first]

  n_rows <- length(enh.gr)
  row_to_gene <- rep(NA_character_, n_rows)
  row_to_gene[qh] <- as.character(S4Vectors::mcols(gh.gr)$GeneID[sh])

  keep <- !is.na(row_to_gene)
  if (!any(keep)) stop("After mapping, no enhancer rows had a GeneHancer match.")

  # subset matrix to mapped rows (sparse or dense)
  if (is_sparse) {
    X <- enhancer.exp[keep, , drop = FALSE]
  } else {
    X <- enhancer.exp[keep, , drop = FALSE]
  }
  row_to_gene <- row_to_gene[keep]

  if (!collapse.rows) {
    # one-to-one required
    if (any(duplicated(row_to_gene))) {
      stop("Duplicated GeneHancer IDs after mapping; set collapse.rows=TRUE to aggregate.")
    }
    if (is_sparse) {
      rownames(X) <- row_to_gene
      return(X)
    } else {
      out <- as.data.frame(X, check.names = FALSE)
      rownames(out) <- row_to_gene
      return(out)
    }
  }

  # --- collapse rows by gene
  ug <- unique(row_to_gene)
  gidx <- match(row_to_gene, ug)                  # 1..G
  G <- length(ug)
  p <- ncol(X)
  cn <- colnames(X)

  # helper for trimmed mean with lots of zeros (no dense materialization)
  .trimmed_mean_zero_aware <- function(v_nonzero, z_zeros, trim) {
    m <- length(v_nonzero) + z_zeros
    if (m == 0L) return(NA_real_)
    k <- floor(trim * m)
    # sort only the nonzeros observed
    if (length(v_nonzero)) {
      v <- sort(v_nonzero, method = "quick")
    } else {
      v <- numeric(0)
    }
    # remove lower tail
    lower_from_zero <- min(k, z_zeros)
    lower_from_v    <- k - lower_from_zero
    if (lower_from_v > 0L) {
      if (lower_from_v >= length(v)) {
        v_after_lower <- numeric(0)
      } else {
        v_after_lower <- v[(lower_from_v + 1L):length(v)]
      }
    } else {
      v_after_lower <- v
    }
    z_after_lower <- z_zeros - lower_from_zero
    # remove upper tail (only affects nonzeros)
    upper_from_v <- min(k, length(v_after_lower))
    if (upper_from_v > 0L) {
      if (upper_from_v < length(v_after_lower)) {
        v_after_both <- v_after_lower[1L:(length(v_after_lower) - upper_from_v)]
      } else {
        v_after_both <- numeric(0)
      }
    } else {
      v_after_both <- v_after_lower
    }
    n_rem <- z_after_lower + length(v_after_both)
    if (n_rem == 0L) return(NA_real_)
    sum(v_after_both) / n_rem
  }

  if (is_sparse && agg %in% c("sum","mean")) {
    # Build sparse row->group indicator (one 1 per row)
    M <- Matrix::sparseMatrix(
      i = seq_len(nrow(X)),
      j = gidx,
      x = 1,
      dims = c(nrow(X), G)
    )
    # sums by group: (G x p)
    agg_mat <- Matrix::t(M) %*% X
    if (agg == "mean") {
      counts <- as.numeric(tabulate(gidx, nbins = G))
      D <- Matrix::Diagonal(x = 1 / counts)
      agg_mat <- D %*% agg_mat
    }
    rownames(agg_mat) <- ug
    colnames(agg_mat) <- cn
    return(agg_mat) # dgCMatrix
  }

  if (!is_sparse && agg %in% c("sum","mean")) {
    # dense fast path
    counts <- as.numeric(tabulate(gidx, nbins = G))
    # build indicator sparsely to keep memory lower than model.matrix
    M <- Matrix::sparseMatrix(i = seq_len(nrow(X)), j = gidx, x = 1, dims = c(nrow(X), G))
    agg_mat <- as.matrix(Matrix::t(M) %*% Matrix::Matrix(X, sparse = TRUE))
    if (agg == "mean") {
      agg_mat <- agg_mat / counts
    }
    rownames(agg_mat) <- ug
    colnames(agg_mat) <- cn
    return(agg_mat)
  }

  # --- trimmed_mean path (works for both sparse and dense X)
  # For sparse X (dgCMatrix): iterate by column, group observed nonzeros, stay sparse on output.
  tol <- 1e-12
  if (is_sparse) {
    Xc <- X
    # precompute group sizes (total loci per gene)
    grp_sizes <- as.numeric(tabulate(gidx, nbins = G))

    I <- integer(0); J <- integer(0); V <- numeric(0)   # triplets for dgTMatrix
    pslot <- Xc@p; islot <- Xc@i; xslot <- Xc@x
    for (j in seq_len(p)) {
      # column j nonzeros are k = (p[j] + 1) : p[j+1]
      k1 <- pslot[j] + 1L; k2 <- pslot[j + 1L]
      if (k2 < k1) next
      rr  <- islot[k1:k2] + 1L       # row indices (1-based)
      val <- xslot[k1:k2]
      if (!length(rr)) next
      gg  <- gidx[rr]                # group for each nonzero

      ord <- order(gg)
      gg  <- gg[ord]
      val <- val[ord]

      # run-length by group
      starts <- c(1L, which(diff(gg) != 0L) + 1L)
      ends   <- c(starts[-1L] - 1L, length(gg))

      for (a in seq_along(starts)) {
        g <- gg[starts[a]]
        v <- val[starts[a]:ends[a]]
        z <- grp_sizes[g] - length(v)
        tm <- .trimmed_mean_zero_aware(v, z, trim)
        if (!is.na(tm) && abs(tm) > tol) {
          I <- c(I, g); J <- c(J, j); V <- c(V, tm)
        }
      }
    }
    if (length(V) == 0L) {
      out <- Matrix::Matrix(0, nrow = G, ncol = p, sparse = TRUE)
    } else {
      out <- Matrix::sparseMatrix(i = I, j = J, x = V, dims = c(G, p))
    }
    rownames(out) <- ug
    colnames(out) <- cn
    if (!return_sparse) out <- as.matrix(out)
    return(out)
  } else {
    # dense + trimmed mean (will return dense)
    counts <- as.numeric(tabulate(gidx, nbins = G))
    out <- matrix(0, nrow = G, ncol = p)
    colnames(out) <- colnames(X); rownames(out) <- ug
    for (j in seq_len(p)) {
      colv <- X[, j]
      # split by group without copying huge lists:
      # use order + runs to slice
      o <- order(gidx)
      gg <- gidx[o]; xv <- colv[o]
      starts <- c(1L, which(diff(gg) != 0L) + 1L)
      ends   <- c(starts[-1L] - 1L, length(gg))
      for (a in seq_along(starts)) {
        g <- gg[starts[a]]
        v <- xv[starts[a]:ends[a]]
        # zeros are explicit zeros in dense vector
        z <- counts[g] - sum(v != 0)
        tm <- .trimmed_mean_zero_aware(v[v != 0], z, trim)
        out[g, j] <- if (is.na(tm)) 0 else tm
      }
    }
    return(out)
  }
}

# Backward-compatible alias (internal)
exp.loci2Ghid_sparse <- exp_loci2Ghid_sparse

#' Map Locus-Indexed Enhancer Expression to GeneHancer IDs
#'
#' Takes an enhancer expression matrix whose rownames are genomic loci
#' (e.g. \code{"chr1:12345-12500"}) and maps each locus to GeneHancer
#' regions provided in a SAF-like table (columns: \code{Chr, Start, End, GeneID}).
#' Optionally collapses rows that map to the same GeneHancer ID using a
#' trimmed mean (10% trim).
#'
#' @param enhancer.exp A numeric matrix or data.frame of enhancer expression
#'   (rows = loci, cols = samples). Row names must be genomic coordinates in the
#'   form \code{chr:start-end}.
#' @param GH.saf A data.frame with at least the columns \code{Chr}, \code{Start},
#'   \code{End}, and \code{GeneID} describing GeneHancer regions.
#' @param collapse.rows Logical; if \code{TRUE} (default), rows that map to the
#'   same \code{GeneID} are collapsed via trimmed mean (\code{trim = 0.1}).
#'
#' @return A numeric data.frame:
#' \itemize{
#'   \item If \code{collapse.rows = TRUE}: rownames are \code{GeneID}, columns are samples.
#'   \item If \code{collapse.rows = FALSE}: rownames are \code{GeneID} (must be unique),
#'         columns are samples; errors if duplicates would occur.
#' }
#'
#' @details
#' Many loci can overlap the same GeneHancer region. With \code{collapse.rows = TRUE},
#' values for all loci mapping to the same \code{GeneID} are aggregated per sample using
#' \code{mean(x, trim = 0.1)} for robustness. With \code{collapse.rows = FALSE}, the
#' function requires a one-to-one mapping between input loci and \code{GeneID}.
#'
#' @examples
#' \dontrun{
#' # enhancer.exp rownames like "chr1:1000-1200"
#' res <- exp.loci2Ghid(enhancer.exp, GH.saf, collapse.rows = TRUE)
#' }
#'
#' @importFrom data.table data.table as.data.table tstrsplit
#' @importFrom GenomicRanges GRanges findOverlaps mcols seqnames start end
#' @importFrom IRanges IRanges
#' @importFrom S4Vectors queryHits subjectHits
#' @export exp_loci2Ghid
#' @export exp.loci2Ghid
exp_loci2Ghid <- function(enhancer.exp, GH.saf, collapse.rows = TRUE) {
  # ---- Input checks ---------------------------------------------------------
  if (is.null(rownames(enhancer.exp)) || anyNA(rownames(enhancer.exp))) {
    stop("`enhancer.exp` must have non-NA rownames in the form 'chr:start-end'.")
  }
  req_cols <- c("Chr", "Start", "End", "GeneID")
  miss <- setdiff(req_cols, colnames(GH.saf))
  if (length(miss)) {
    stop("`GH.saf` is missing required columns: ", paste(miss, collapse = ", "))
  }

  # Coerce enhancer.exp to data.frame for safe cbind later
  # Keep numeric storage; do not convert strings to factors
  if (is.matrix(enhancer.exp)) {
    enhancer_df <- as.data.frame(enhancer.exp, stringsAsFactors = FALSE, check.names = FALSE)
  } else if (is.data.frame(enhancer.exp)) {
    enhancer_df <- enhancer.exp
  } else {
    stop("`enhancer.exp` must be a matrix or data.frame.")
  }

  # ---- Build enhancer GRanges from rowname coordinates ----------------------
  loci.dt <- data.table::data.table(coord = rownames(enhancer_df))
  # split "chr:start-end" robustly
  loci.dt[, c("seqnames", "start", "end") := data.table::tstrsplit(coord, "[\\:\\-]", type.convert = TRUE)]
  if (anyNA(loci.dt$seqnames) || anyNA(loci.dt$start) || anyNA(loci.dt$end)) {
    bad <- loci.dt$coord[is.na(loci.dt$seqnames) | is.na(loci.dt$start) | is.na(loci.dt$end)]
    stop("Malformed locus coordinates in rownames (expected 'chr:start-end'). Examples: ",
         paste(utils::head(bad, 3), collapse = ", "), if (length(bad) > 3) " ...")
  }

  enhancer.gr <- GenomicRanges::GRanges(
    seqnames = loci.dt$seqnames,
    ranges   = IRanges::IRanges(start = loci.dt$start, end = loci.dt$end),
    strand   = "*",
    GeneID   = loci.dt$coord # store the original locus string
  )

  gh.gr <- GenomicRanges::GRanges(
    seqnames = GH.saf$Chr,
    ranges   = IRanges::IRanges(start = as.integer(GH.saf$Start), end = as.integer(GH.saf$End)),
    strand   = "*",
    GeneID   = GH.saf$GeneID
  )

  # ---- Overlap + mapping ----------------------------------------------------
  hits <- GenomicRanges::findOverlaps(enhancer.gr, gh.gr)
  if (length(hits) == 0L) {
    stop("No overlaps found between enhancer loci and GH.saf regions.")
  }

  overlapping_enhancers <- enhancer.gr[S4Vectors::queryHits(hits)]
  overlapping_gh        <- gh.gr[S4Vectors::subjectHits(hits)]

  mapped_results <- data.frame(
    EnhancerID     = as.character(S4Vectors::mcols(overlapping_enhancers)$GeneID),
    GeneHancerID   = as.character(S4Vectors::mcols(overlapping_gh)$GeneID),
    EnhancerChr    = as.character(GenomicRanges::seqnames(overlapping_enhancers)),
    EnhancerStart  = GenomicRanges::start(overlapping_enhancers),
    EnhancerEnd    = GenomicRanges::end(overlapping_enhancers),
    GeneHancerChr  = as.character(GenomicRanges::seqnames(overlapping_gh)),
    GeneHancerStart= GenomicRanges::start(overlapping_gh),
    GeneHancerEnd  = GenomicRanges::end(overlapping_gh),
    stringsAsFactors = FALSE
  )

  # Map each enhancer row (by locus rowname) to its first matching GeneHancerID
  # (If multiple overlaps per locus, the first is chosen prior to optional collapsing.)
  Gene <- mapped_results$GeneHancerID[match(rownames(enhancer_df), mapped_results$EnhancerID)]
  enhancer_df$Gene <- Gene
  enhancer_df <- enhancer_df[!is.na(enhancer_df$Gene), , drop = FALSE]

  if (!nrow(enhancer_df)) {
    stop("After mapping, no enhancer rows had a GeneHancer match.")
  }

  # ---- Collapse or ensure uniqueness ----------------------------------------
  if (isTRUE(collapse.rows)) {
    dt <- data.table::as.data.table(enhancer_df)
    # group by Gene; apply trimmed mean to numeric columns only
    num_cols <- vapply(dt, is.numeric, logical(1))
    if (!any(num_cols)) stop("No numeric expression columns detected in `enhancer.exp`.")
    # Keep only numeric sample columns in .SD
    dt_numeric <- dt[, c("Gene", names(dt)[num_cols]), with = FALSE]
    collapsed <- dt_numeric[, lapply(.SD, function(x) mean(as.numeric(x), trim = 0.1)), by = Gene]
    out <- data.frame(row.names = collapsed$Gene,
                      collapsed[, -1, drop = FALSE],
                      stringsAsFactors = FALSE, check.names = FALSE)
    return(out)
  } else {
    # Require one-to-one locus -> Gene mapping (no duplicate Gene IDs after mapping)
    if (any(duplicated(enhancer_df$Gene))) {
      stop("There are duplicated GeneHancer IDs after mapping; set collapse.rows = TRUE to aggregate.")
    }
    rn <- enhancer_df$Gene
    enhancer_df$Gene <- NULL
    out <- data.frame(row.names = rn, enhancer_df, stringsAsFactors = FALSE, check.names = FALSE)
    return(out)
  }
}

# Backward-compatible alias (internal)
exp.loci2Ghid <- exp_loci2Ghid

#' Map SAF-Format Loci to GeneHancer IDs
#'
#' Given two SAF-like data frames—one for enhancer loci and one for GeneHancer
#' regions—return the overlap mapping between enhancer loci and GeneHancer IDs.
#'
#' @param enhancer.saf A data.frame with columns \code{Chr}, \code{Start}, \code{End}, \code{GeneID}
#'   describing enhancer loci in SAF format.
#' @param GH.saf A data.frame with columns \code{Chr}, \code{Start}, \code{End}, \code{GeneID}
#'   describing GeneHancer regions in SAF format.
#'
#' @return A data.frame with one row per overlap containing:
#' \itemize{
#'   \item \code{EnhancerID}, \code{EnhancerChr}, \code{EnhancerStart}, \code{EnhancerEnd}
#'   \item \code{GeneHancerID}, \code{GeneHancerChr}, \code{GeneHancerStart}, \code{GeneHancerEnd}
#' }
#' If no overlaps are found, the function errors with an informative message.
#'
#' @examples
#' \dontrun{
#' enh <- data.frame(
#'   Chr = c("chr1","chr1","chr2"),
#'   Start = c(100, 500, 1000),
#'   End = c(200, 650, 1200),
#'   GeneID = c("loc1","loc2","loc3")
#' )
#' gh <- data.frame(
#'   Chr = c("chr1","chr2"),
#'   Start = c(150, 900),
#'   End = c(700, 1100),
#'   GeneID = c("GH001","GH002")
#' )
#' map <- safloci2Ghid(enh, gh)
#' }
#'
#' @importFrom GenomicRanges GRanges findOverlaps mcols seqnames start end
#' @importFrom IRanges IRanges
#' @importFrom S4Vectors queryHits subjectHits
#' @export
safloci2Ghid <- function(enhancer.saf, GH.saf) {
  # ---- input checks ----
  req_cols <- c("Chr", "Start", "End", "GeneID")
  miss_enh <- setdiff(req_cols, colnames(enhancer.saf))
  miss_gh  <- setdiff(req_cols, colnames(GH.saf))
  if (length(miss_enh)) stop("`enhancer.saf` is missing columns: ", paste(miss_enh, collapse = ", "))
  if (length(miss_gh))  stop("`GH.saf` is missing columns: ", paste(miss_gh, collapse = ", "))

  # Coerce starts/ends to integer (avoids accidental factors/characters)
  enhancer.saf$Start <- as.integer(enhancer.saf$Start)
  enhancer.saf$End   <- as.integer(enhancer.saf$End)
  GH.saf$Start       <- as.integer(GH.saf$Start)
  GH.saf$End         <- as.integer(GH.saf$End)

  # ---- build GRanges ----
  enhancer.gr <- GenomicRanges::GRanges(
    seqnames = enhancer.saf$Chr,
    ranges   = IRanges::IRanges(start = enhancer.saf$Start, end = enhancer.saf$End),
    strand   = "*",
    GeneID   = enhancer.saf$GeneID
  )
  gh.gr <- GenomicRanges::GRanges(
    seqnames = GH.saf$Chr,
    ranges   = IRanges::IRanges(start = GH.saf$Start, end = GH.saf$End),
    strand   = "*",
    GeneID   = GH.saf$GeneID
  )

  # ---- overlaps ----
  hits <- GenomicRanges::findOverlaps(enhancer.gr, gh.gr)
  if (length(hits) == 0L) {
    stop("No overlaps found between enhancer.saf and GH.saf.")
  }

  overlapping_enhancers <- enhancer.gr[S4Vectors::queryHits(hits)]
  overlapping_gh        <- gh.gr[S4Vectors::subjectHits(hits)]

  # ---- assemble mapping ----
  mapped_results <- data.frame(
    EnhancerID       = as.character(GenomicRanges::mcols(overlapping_enhancers)$GeneID),
    GeneHancerID     = as.character(GenomicRanges::mcols(overlapping_gh)$GeneID),
    EnhancerChr      = as.character(GenomicRanges::seqnames(overlapping_enhancers)),
    EnhancerStart    = GenomicRanges::start(overlapping_enhancers),
    EnhancerEnd      = GenomicRanges::end(overlapping_enhancers),
    GeneHancerChr    = as.character(GenomicRanges::seqnames(overlapping_gh)),
    GeneHancerStart  = GenomicRanges::start(overlapping_gh),
    GeneHancerEnd    = GenomicRanges::end(overlapping_gh),
    stringsAsFactors = FALSE
  )

  mapped_results
}

#' Filter Intragenic GeneHancers
#'
#' Filters out intragenic GeneHancers by excluding those overlapping with extended exon regions,
#' and optionally filters out promoter elements. Both \code{GeneHancer.file} and \code{GeneCode.file}
#' must be based on the same genome build.
#'
#' @param GeneHancer.file A character string specifying the path to the GeneHancer annotation file.
#' @param GeneCode.file A character string specifying the path to the GTF file (gene annotations).
#' @param extend.bp An integer specifying the number of base pairs to extend exon regions (default is 0).
#' @param filter.promoter Logical. If TRUE, filters out GeneHancer entries classified as promoters (default is TRUE).
#' @param filter.exon Logical. If TRUE, filters out GeneHancer entries that overlap with extended exon regions (default is TRUE).
#'
#' @return A data frame containing the filtered GeneHancer annotations.
#'
#' @importFrom data.table fread
#' @importFrom rtracklayer import
#' @importFrom GenomicRanges GRanges findOverlaps mcols seqnames start end strand
#' @importFrom IRanges IRanges
#'
#' @export
filter_geneHancer <- function(GeneHancer.file, GeneCode.file, extend.bp = 0, filter.promoter = TRUE, filter.exon = TRUE) {
  GeneHancer <- data.table::fread(GeneHancer.file, stringsAsFactors = FALSE, check.names = FALSE, header = TRUE, sep = "\t")
  if (filter.exon) {
    gtf_data <- rtracklayer::import(GeneCode.file)
    exon_features <- gtf_data[gtf_data$type == "exon"]
    exon_locations <- data.frame(
      exon_id = GenomicRanges::mcols(exon_features)$exon_id,
      gene_id = GenomicRanges::mcols(exon_features)$gene_id,
      seqnames = GenomicRanges::seqnames(exon_features),
      start = GenomicRanges::start(exon_features),
      end = GenomicRanges::end(exon_features),
      strand = GenomicRanges::strand(exon_features))
    # Extend the exon regions by the specified number of base pairs
    exon_gr <- GenomicRanges::GRanges(
      seqnames = exon_locations$seqnames,
      ranges = IRanges::IRanges(
        start = pmax(0, exon_locations$start - extend.bp),  # Ensure no negative coordinates
        end = exon_locations$end + extend.bp))
    # Create GRanges objects for GeneHancer
    genehancer_gr <- GenomicRanges::GRanges(
      seqnames = paste0("chr", GeneHancer$chr),
      ranges = IRanges::IRanges(start = GeneHancer$element_start, end = GeneHancer$element_end))
    # Find overlaps between GeneHancer and exon locations
    overlaps <- GenomicRanges::findOverlaps(genehancer_gr, exon_gr)
    # Exclude GeneHancer entries that overlap with extended exon locations
    genehancer_filtered <- GeneHancer[-unique(S4Vectors::queryHits(overlaps)), ]
  } else {
    genehancer_filtered <- GeneHancer
  }
  if (filter.promoter) {
    genehancer_filtered <- genehancer_filtered[!grepl("Promoter", genehancer_filtered$regulatory_element_type, ignore.case = TRUE), ]
  }
  return(genehancer_filtered)
}

#' Preprocess Enhancer Counts (GeneHancer-based + scenario-specific low-expression filters)
#'
#' Keeps only enhancers present in GeneHancer after removing promoters and exon overlaps,
#' then applies a low-expression filter tailored to either (i) pairwise designs with group labels
#' or (ii) collections without group information. Optionally returns TMM-normalized CPM values
#' using edgeR.
#'
#' @param enhancer.counts Numeric matrix of raw counts; rows = enhancers (GHid), cols = samples.
#' @param filter.low Logical; if TRUE apply the scenario-specific low-expression filter (default FALSE).
#' @param detect.rate Numeric in (0,1); required for the ungrouped filter. Represents the minimum fraction
#'   of samples that must reach \code{detect.count} reads.
#' @param GeneHancer.file Path to GeneHancer annotation file.
#' @param GeneCode.file   Path to GTF file with gene annotations.
#' @param meta_data Optional data.frame with sample-level metadata (rownames must match column names of
#'   \code{enhancer.counts}).
#' @param group.col Character scalar naming the column in \code{meta_data} that encodes group membership
#'   for pairwise designs. Ignored if \code{group} is supplied.
#' @param group Optional factor/character vector of group labels aligned to the columns of
#'   \code{enhancer.counts}. Overrides \code{group.col} if provided.
#' @param detect.count Integer threshold of raw counts used for the detect-rate filter in the ungrouped
#'   scenario. Defaults to 1.
#' @param min.samples Optional integer with the minimum number of samples that must pass \code{detect.count}
#'   when using the detect-rate filter. If NULL, it is derived from \code{detect.rate}.
#' @param CPM_tMM Logical; if TRUE, compute and return TMM-normalized CPM (non-log) alongside the filtered
#'   counts.
#' @param ... Optional arguments. The legacy argument name \code{enhancer.matrix} is accepted via \code{...}
#'   for backward compatibility.
#'
#' @return If \code{CPM_tMM = TRUE}, a list with elements \code{counts} (filtered counts) and \code{cpm_tmm}
#'   (TMM-normalized CPM matrix). Otherwise, returns the filtered counts matrix.
#' @export
preprocess_enhancerMatrix <- function(
    enhancer.counts,
    filter.low = FALSE,
    detect.rate = NULL,
    GeneHancer.file,
    GeneCode.file,
    meta_data = NULL,
    group.col = NULL,
    group = NULL,
    detect.count = 1L,
    min.samples = NULL,
    CPM_tMM = FALSE,
    ...
){
  dots <- list(...)
  if (missing(enhancer.counts) && "enhancer.matrix" %in% names(dots)) {
    enhancer.counts <- dots$enhancer.matrix
    dots$enhancer.matrix <- NULL
  }
  if (length(dots)) {
    warning("Unused arguments in preprocess_enhancerMatrix: ",
            paste(names(dots), collapse = ", "))
  }
  if (missing(enhancer.counts) || is.null(enhancer.counts)) {
    stop("`enhancer.counts` must be provided.")
  }

  # 1) Filter GeneHancer (exclude promoters and exon overlaps)
  gh_filtered <- filter_geneHancer(
    GeneHancer.file = GeneHancer.file,
    GeneCode.file   = GeneCode.file,
    filter.promoter = TRUE,
    filter.exon     = TRUE
  )

  counts <- as.matrix(enhancer.counts)
  rownames(counts) <- rownames(enhancer.counts)
  colnames(counts) <- colnames(enhancer.counts)

  counts <- counts[rownames(counts) %in% gh_filtered$GHid, , drop = FALSE]
  if (!nrow(counts)) return(counts)

  # harmonise metadata / group info if provided
  if (!is.null(group)) {
    if (length(group) != ncol(counts)) {
      stop("`group` must have the same length as the number of samples (columns) in enhancer.counts.")
    }
    group <- droplevels(factor(group))
  } else if (!is.null(meta_data) && !is.null(group.col)) {
    if (!group.col %in% colnames(meta_data)) {
      stop("`group.col` (", group.col, ") not found in meta_data.")
    }
    if (!all(colnames(counts) %in% rownames(meta_data))) {
      missing_samples <- setdiff(colnames(counts), rownames(meta_data))
      stop("meta_data is missing rows for samples: ", paste(missing_samples, collapse = ", "))
    }
    group <- droplevels(factor(meta_data[colnames(counts), group.col]))
  } else {
    group <- NULL
  }

  # 2) Scenario-specific low-expression filtering
  if (isTRUE(filter.low)) {
    if (!is.null(group)) {
      suppressMessages({
        y <- edgeR::DGEList(counts = counts, group = group)
      })
      keep <- edgeR::filterByExpr(y, group = group)
      if (!any(keep)) {
        stop("No enhancers passed edgeR::filterByExpr; reconsider group labels or upstream filtering.")
      }
      y <- y[keep, , keep.lib.sizes = FALSE]
      counts <- y$counts
      y_filtered <- y
    } else {
      if (is.null(detect.rate) || !is.numeric(detect.rate) || detect.rate <= 0 || detect.rate >= 1) {
        stop("When filter.low=TRUE without group information, provide detect.rate in (0,1), e.g., 0.05 or 0.10.")
      }
      if (!is.numeric(detect.count) || detect.count < 0) {
        stop("`detect.count` must be a non-negative numeric scalar.")
      }
      if (is.null(min.samples)) {
        min.samples <- ceiling(detect.rate * ncol(counts))
      }
      min.samples <- as.integer(min.samples)
      min.samples <- max(1L, min.samples)
      keep <- rowSums(counts >= detect.count, na.rm = TRUE) >= min.samples
      if (!any(keep)) {
        stop("No enhancers passed the detect-rate filter; adjust detect.rate or detect.count.")
      }
      counts <- counts[keep, , drop = FALSE]
      y_filtered <- NULL
    }
  } else {
    y_filtered <- NULL
  }

  if (isTRUE(CPM_tMM)) {
    if (!is.null(y_filtered)) {
      y_norm <- edgeR::calcNormFactors(y_filtered, method = "TMM")
      cpm_tmm <- edgeR::cpm(y_norm, normalized.lib.sizes = TRUE, log = FALSE)
      counts_out <- y_norm$counts
    } else {
      y_norm <- edgeR::DGEList(counts = counts)
      y_norm <- edgeR::calcNormFactors(y_norm, method = "TMM")
      cpm_tmm <- edgeR::cpm(y_norm, normalized.lib.sizes = TRUE, log = FALSE)
      counts_out <- y_norm$counts
    }
    return(list(
      counts = counts_out,
      cpm_tmm = cpm_tmm
    ))
  }

  counts
}

#' Write Gene Set GMT File
#'
#' Writes a gene set list to a GMT (Gene Matrix Transposed) file. If the specified file already exists,
#' it will be deleted before writing the new file.
#'
#' @param geneset.list A named list where each element is a vector of gene symbols (or identifiers). The names
#'   of the list represent the gene set names.
#' @param file.name A character string specifying the path and filename of the output GMT file.
#' @param default_desc A character string used as the description column when none is supplied
#'   in the gene set name. Default is \code{"."}.
#'
#' @return No return value; the function writes the GMT file to disk.
#'
#' @details
#' The function writes a header line followed by each gene set in GMT format. For each gene set, it splits the gene set name
#' by the last colon (if present) and writes the gene set name, a description (if any), and the list of genes.
#' When no description is provided, the `default_desc` placeholder is used.
#'
#' @export
write_gmt <- function(geneset.list, file.name, default_desc = ".") {
  if (file.exists(file.name)) {
    message(paste(file.name, "exists, deleting..."))
    file.remove(file.name)
  }
  write("#GMT file written from a genotypelist", file=file.name, sep="\t",append=TRUE, ncolumns=1)
  for (i in seq_along(geneset.list)) {
    genotype <- geneset.list[[i]]
    gname <- names(geneset.list)[i]
    parts <- strsplit(gname, ":(?=[^:]+$)", perl = TRUE)[[1]]
    if (length(parts) == 1L) {
      set_name <- parts
      desc <- default_desc
    } else {
      set_name <- parts[1L]
      desc <- parts[2L]
      if (!nzchar(desc)) {
        desc <- default_desc
      }
    }
    line <- c(set_name, desc, genotype)
    write(line, file = file.name, sep = "\t", append = TRUE, ncolumns = length(line))
  }
}
