
#' pre-define a group of normal cells with GMM.
#'
#' @param CNA.mat smoothed data matrix; genes in rows; cell names in columns.
#' @param max.normal find the first number diploid cells to save efforts.
#' @param mu.cut diploid baseline cutoff.
#' @param Nfraq.cut minimal fractoins of genomes with CNAs.
#'
#' @return 1) predefined diploid cell names; 2) clustering results; 3) inferred baseline.
#'
#' @examples
#' test.gmm <- baseline.GMM(CNA.mat=smooth.com, max.normal=30, mu.cut=0.05, Nfraq.cut=0.99)
#'
#' test.gmm.cells <- test.bnc$preN
#' @export
baseline.GMM <- function(CNA.mat, max.normal=5, mu.cut=0.05, Nfraq.cut=0.99, RE.before=basa, n.cores=1){

     N.normal <-NULL
     for(m in 1:ncol(CNA.mat)){

      print(paste("cell: ", m, sep=""))
      sam <- CNA.mat[, m]
      sg <- max(c(0.05, 0.5*sd(sam)));
      GM3 <- mixtools::normalmixEM(sam, lambda = rep(1,3)/3, mu = c(-0.2, 0, 0.2),sigma = sg,arbvar=FALSE,ECM=FALSE,maxit=5000)

      ###decide normal or tumor
      s <- sum(abs(GM3$mu)<=mu.cut)
      print(s)
      if(s>=1){
        frq <- sum(GM3$lambda[which(abs(GM3$mu)<=mu.cut)])

        if(frq> Nfraq.cut){
          pred <- "diploid"
        }else{pred<-"aneuploid"}

      }else {pred <- "aneuploid"}
    #  print(paste("pred: ", pred, sep=""))
       N.normal<- c(N.normal,pred)
      print(N.normal)
      if(sum(N.normal=="diploid")>=max.normal){break}
      m<- m+1
    }

    names(N.normal) <- colnames(CNA.mat)[1:length(N.normal)]
    preN <- names(N.normal)[which(N.normal=="diploid")]

    d <- parallelDist::parDist(t(CNA.mat), threads = n.cores) ##use smooth and segmented data to detect intra-normal cells
    km <- 6
    fit <- hclust(d, method="ward.D2")
    ct <- cutree(fit, k=km)


    if(length(preN) >2){
      WNS <- ""
      basel <- apply(CNA.mat[, which(colnames(CNA.mat) %in% preN)], 1, mean)

      RE <- list(basel, WNS, preN, ct)
      names(RE) <- c("basel", "WNS", "preN", "cl")
      return(RE)
    }else{
      return(RE.before) ##found this bug
    }

}


baseline.norm.cl <- function(norm.mat.smooth, min.cells=5, n.cores=n.cores){

  d <- parallelDist::parDist(t(norm.mat.smooth), threads = n.cores) ##use smooth and segmented data to detect intra-normal cells
  km <- 6
  fit <- hclust(d, method="ward.D2")
  ct <- cutree(fit, k=km)

  while(!all(table(ct)>min.cells)){
    km <- km -1
    ct <- cutree(fit, k=km)
    if(km==2){
      break
    }
  }

  SDM <-NULL
  SSD <-NULL
  for(i in min(ct):max(ct)){

    data.c <- apply(norm.mat.smooth[, which(ct==i)],1, median)
    sx <- max(c(0.05, 0.5*sd(data.c)))
    GM3 <- mixtools::normalmixEM(data.c, lambda = rep(1,3)/3, mu = c(-0.2, 0, 0.2), sigma = sx,arbvar=FALSE,ECM=FALSE,maxit=5000)
    SDM <- c(SDM, GM3$sigma[1])
    SSD <- c(SSD, sd(data.c))
       i <- i+1
      }

  wn <- mean(cluster::silhouette(cutree(fit, k=2), d)[, "sil_width"])

 PDt <- pf(max(SDM)^2/min(SDM)^2, nrow(norm.mat.smooth), nrow(norm.mat.smooth), lower.tail = FALSE)

  if(wn <= 0.15|(!all(table(ct)>min.cells))| PDt > 0.05){
    WNS <- "unclassified.prediction"
    print("low confidence in classification")
  }else {
    WNS <- ""
  }
    basel <- apply(norm.mat.smooth[, which(ct %in% which(SDM==min(SDM)))], 1, median)
    preN <- colnames(norm.mat.smooth)[which(ct %in% which(SDM==min(SDM)))]
    
  ### return both baseline and warning message
  RE <- list(basel, WNS, preN, ct)
  names(RE) <- c("basel", "WNS", "preN", "cl")
  return(RE)
    }


dlm.sm <- function(c){
    model <- dlm::dlmModPoly(order=1, dV=0.16, dW=0.001)
    x <- dlm::dlmSmooth(norm.mat[, c], model)$s
    x<- x[2:length(x)]
    x <- x-mean(x)
  }

test <- read.table('/home/jbonet/Desktop/copyVAE/data_and_more/bined_expressed_cell_trans_clean_nonames.csv', sep='\t', header=TRUE)


print("Normalizing matrix...")
rawmat3 <- data.matrix(test)
# rawmat3 <- data.matrix(test[, 8:ncol(test)])
norm.mat<- log(sqrt(rawmat3)+sqrt(rawmat3+1))
norm.mat<- apply(norm.mat,2,function(x)(x <- x-mean(x)))
colnames(norm.mat) <-  colnames(rawmat3)

print("Dynamic Linear Model in process...")
test.mc <-parallel::mclapply(1:ncol(norm.mat), dlm.sm, mc.cores = 4)
norm.mat.smooth <- matrix(unlist(test.mc), ncol = ncol(norm.mat), byrow = FALSE)
colnames(norm.mat.smooth) <- colnames(norm.mat)

print("step 4: measuring baselines ...")
basa <- baseline.norm.cl(norm.mat.smooth=norm.mat.smooth, min.cells=5, n.cores=4)
basel <- basa$basel
WNS <- basa$WNS
preN <- basa$preN
CL <- basa$cl

if (WNS =="unclassified.prediction"){
  basa <- baseline.GMM(CNA.mat=norm.mat.smooth, max.normal=5, mu.cut=0.05, Nfraq.cut=0.99,RE.before=basa,n.cores=4)
  basel <-basa$basel
  WNS <- basa$WNS
  preN <- basa$preN

}

write.table(preN, 'normal_cells.txt', sep="\t", row.names=FALSE, quote=F)
save(preN, file='normal_cells_2.rdata')


print("Done!")