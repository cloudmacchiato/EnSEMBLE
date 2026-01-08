#' Compute VIPER Enhancer Scores
#'
#' Computes VIPER enhancer scores by creating a regulon object from enhancer sets
#' and applying the VIPER algorithm to the provided CPM matrix.
#'
#' @param cpmMatrix A numeric matrix representing counts per million (CPM) expression values,
#'   where rows represent enhancers and columns represent samples.
#' @param enhancersets A list in which each element is a vector of enhancers for a specific context.
#' @param cores Integer. Number of cores to use for parallel computation (default is 1). Parallel computation can significantly reduce the time for pleiotropy (might need to adjust the cores if face certain errors).
#' @param pleiotropy Logical. Whether to apply pleiotropy correction (default is TRUE).
#' @param pleiotropyArgs A list of parameters for pleiotropy correction. Default is
#'   \code{list(regulators = 0.05, shadow = 0.05, targets = 10, penalty = 20, method = "adaptive")}.
#'
#' @return A VIPER score matrix as returned by the \code{viper} function from the VIPER package.
#'
#' @details
#' The function constructs a regulon object by assigning a weight of 1 and equal confidence to each
#' target gene in the provided enhancer sets. It then computes the VIPER score using \code{viper::viper()}.
#' Pleiotropy correction is highly recommended if some enhancer sets share large number of enhancers.
#'
#' @import viper
#'
#' @export
viper_enhancer_score <- function(cpmMatrix, enhancersets, cores = 1, pleiotropy = TRUE,
                                 pleiotropyArgs = list(regulators = 0.05, shadow = 0.05, targets = 10, penalty = 20, method = "adaptive")) {

  regulon_obj <- lapply(enhancersets, function(targets) {
    list(
      tfmode = setNames(rep(1, length(targets)), targets),  # Weight of 1 for all target genes
      likelihood = rep(1, length(targets))                   # Equal confidence for all targets
    )
  })

  viper_score <- viper::viper(cpmMatrix, regulon_obj, pleiotropy = pleiotropy, cores = cores, pleiotropyArgs = pleiotropyArgs)
  return(viper_score)
}

#' Plot Heatmap for eRNA Matrix or Enhancer Set Score Matrix
#'
#' Generates a heatmap for an eRNA expression or enhancer set score matrix.
#' Optionally filters to keep only the top most variable rows based on a specified quantile.
#'
#' @param enhancer.matrix A numeric matrix or data frame with eRNA expression or enhancer scores.
#'   Row names should represent eRNA identifiers or enhancer setnames. Columns are samples.
#' @param scale A character string specifying the type of scaling to apply.
#'   Options are \code{"row"}, \code{"col"}, or \code{"both"}. Default is \code{"row"}.
#' @param cluster_rows Logical indicating whether to cluster rows in the heatmap.
#'   Default is \code{TRUE}.
#' @param cluster_cols Logical indicating whether to cluster columns in the heatmap.
#'   Default is \code{TRUE}.
#' @param show_rownames Logical indicating whether to display row names in the heatmap.
#'   Default is \code{TRUE}.
#' @param show_colnames Logical indicating whether to display column names in the heatmap.
#'   Default is \code{TRUE}.
#' @param filter.percent Numeric value between 0 and 1.
#'   If provided, only the top \code{filter.percent * 100} percent most variable rows are retained.
#' @param ann_col Optional annotations for columns (a data frame or list). Default is \code{NA}.
#' @param ann_row Optional annotations for rows (a data frame or list). Default is \code{NA}.
#' @param fontsize_row Numeric value for the row font size in the heatmap. Default is \code{3}.
#' @param fontsize_col Numeric value for the column font size in the heatmap. Default is \code{3}.
#'
#' @return A heatmap is generated using the \code{pheatmap} package.
#'
#' @details
#' If \code{filter.percent} is specified, the function converts the input matrix to a data.table,
#' computes the variance for each row (excluding the identifier column), and retains only rows with variance
#' above the specified quantile threshold. The matrix is then scaled according to the \code{scale} argument,
#' and a heatmap is generated with a custom color palette and breakpoints.
#'
#' @import pheatmap
#' @export
ENSEMBLE.heatmap <- function(enhancer.matrix,scale="row",cluster_rows=T,cluster_cols=T,show_rownames=T,show_colnames=T,filter.percent=NA,ann_col=NA,ann_row=NA,fontsize_row = 3,fontsize_col = 3){
  if (!is.na(filter.percent)){
    enhancer.matrix=as.data.table(enhancer.matrix, keep.rownames = "eRNA")
    enhancer.matrix[, row_variance := apply(.SD, 1, var), .SDcols = names(enhancer.matrix)[!grepl("eRNA",names(enhancer.matrix))]]
    # Determine the threshold for the top 10% most variated rows
    threshold <- quantile(enhancer.matrix$row_variance, filter.percent)  # Correct for top 10% (0.9 quantile)
    # Subset the top 10% most variated rows
    enhancer.matrix <- enhancer.matrix[row_variance >= threshold, .SD, .SDcols = names(enhancer.matrix)[!grepl("row_variance",names(enhancer.matrix))]]
    enhancer.matrix=data.frame(enhancer.matrix,stringsAsFactors = F, check.names = F)
    enhancer.matrix <- data.frame(row.names = enhancer.matrix[,"eRNA"],enhancer.matrix[,colnames(enhancer.matrix)!="eRNA"],stringsAsFactors = F, check.names = F)
  }
  enhancer.matrix=as.matrix(enhancer.matrix)
  if (scale=="row"){
    enhancer.matrix=t(scale(t(enhancer.matrix)))
  }else if(scale=="col"){
    enhancer.matrix=scale(enhancer.matrix)
  }else if (scale=="both"){
    enhancer.matrix=t(scale(t(scale(enhancer.matrix))))
  }else{
    enhancer.matrix=enhancer.matrix
  }
  quantile.range <- quantile(enhancer.matrix, probs = seq(0, 1, 0.01))
  paletteLength <- 50
  myColor <- colorRampPalette(c("skyblue", "white", "red"))(paletteLength)
  myBreaks <- c(seq(quantile.range["5%"], 0, length.out=ceiling(paletteLength/2) + 1),
                seq(quantile.range["95%"]/paletteLength, quantile.range["95%"], length.out=floor(paletteLength/2)))
  pheatmap(
    enhancer.matrix,
    cluster_cols = cluster_cols,
    cluster_rows = cluster_rows,
    clustering_method = "ward.D2",  # Use Ward's D2 method for clustering
    main = ifelse(is.na(filter.percent),"Heatmap",paste0("Heatmap of Top ",filter.percent*100,"% Most Variated Rows")),
    show_rownames = show_rownames,         # Disable row names in the heatmap
    show_colnames = show_colnames,
    color=myColor,
    breaks=myBreaks,
    annotation_row=ann_row,
    annotation_col=ann_col,
    fontsize_row = fontsize_row,
    fontsize_col = fontsize_col
  )
}

#' Differential Enhancer Expression (DEE) Analysis using DESeq2
#'
#' Performs differential enhancer expression (DEE) analysis for eRNA using the DESeq2 package.
#' Raw counts, sample metadata, and a contrast vector are required. The contrast vector must be a character vector
#' with exactly three elements: the name of a factor in the design formula, the numerator level (for the fold change),
#' and the denominator level. For example, \code{contrast = c("group", "case", "control")}.
#'
#' @param eRNA.counts A numeric matrix of eRNA counts. Row names should represent enhancer identifiers (GHid)
#'   and column names should represent sample names.
#' @param meta_data A data frame containing metadata for each sample. Row names of \code{meta_data} must match the column
#'   names of \code{eRNA.counts}.
#' @param contrast A character vector with three elements specifying the contrast for the DESeq2 analysis.
#' @param round.counts Logical. If \code{TRUE} (default), the raw counts are rounded to the nearest integer.
#' @param min.countsum Numeric. Minimum total counts required for an enhancer to be retained (default is 10).
#'
#' @return A data frame containing the differential expression results, sorted by adjusted p-value (\code{padj}),
#'   with a column \code{GHid} for enhancer identifiers.
#'
#' @details
#' This function first subsets the eRNA counts matrix to include only the samples present in the metadata.
#' If \code{round.counts} is \code{TRUE}, counts are rounded. A DESeq2 dataset is then created using the design formula
#' \code{~ group}. Enhancers with very low counts (row sums less than \code{min.countsum}) are filtered out before running
#' the DESeq2 pipeline. Finally, the DESeq2 results for the specified contrast are extracted, converted to a data frame,
#' and sorted by adjusted p-values.
#'
#' @importFrom DESeq2 DESeqDataSetFromMatrix counts DESeq results
#' @export
ENSEMBLE.DEE <- function(eRNA.counts, meta_data, contrast, round.counts = TRUE, min.countsum = 10) {
  # Subset the counts matrix to include only the samples in meta_data
  eRNA.counts <- eRNA.counts[, rownames(meta_data)]
  meta_data <- meta_data[,contrast[1],drop=FALSE]
  design_formula <- as.formula(paste("~", contrast[1]))
  # Optionally round the counts to integers
  if (round.counts == TRUE) {
    eRNA.counts <- round(eRNA.counts)
  }
  # Create a DESeq2 dataset using the design formula
  dds <- DESeq2::DESeqDataSetFromMatrix(countData = eRNA.counts,
                                        colData = meta_data,
                                        design = design_formula)
  # Filter out low-count enhancers
  dds <- dds[rowSums(DESeq2::counts(dds)) > min.countsum, ]
  # Run the DESeq2 pipeline
  dds <- DESeq2::DESeq(dds)
  # Extract results for the specified contrast
  res <- DESeq2::results(dds, contrast = contrast)
  # Convert the results to a data frame and sort by adjusted p-value
  results_df <- as.data.frame(res)
  results_df <- results_df[order(results_df$padj), ]
  results_df <- data.frame(GHid = rownames(results_df), results_df, stringsAsFactors = FALSE, check.names = FALSE)
  return(results_df)
}

#' Compute Differential Expression Weights
#'
#' Calculates weights from differential expression results by taking the sign of the test statistic
#' multiplied by the negative log10 of the p-value, and assigns them to the corresponding enhancer identifiers.
#'
#' @param DE_result A data frame output from ENSEMBLE.DEE containing differential expression results. It must include the columns:
#'   \code{stat} (test statistic), \code{pvalue} (p-values), and \code{GHid} (enhancer identifiers).
#'
#' @return A named numeric vector of weights, where each weight is computed as:
#' \deqn{w = sign(stat) \times -\log_{10}(pvalue)}
#'
#' @details
#' This function computes weights for downstream analyses, such as enhancer set enrichment analysis.
#'
#' @export
DE.weights <- function(DE_result){
  p_safe <- pmax(DE_result$pvalue, .Machine$double.xmin)  # ~2.2e-308
  weights <- setNames(sign(DE_result$stat) * -log10(p_safe), DE_result$GHid)
  weights[is.na(weights)] <- 0
  return(weights)
}

