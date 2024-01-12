.libPaths("/home/anran/miniconda3/lib/R/library")
library(glasso)

args <- commandArgs(T)
c_path <- args[1]
invc_path <- args[2]
invg_path <- args[3]
rho1 <- as.numeric(args[4])
rho2 <- as.numeric(args[5])

cmat <- read.table(file = c_path)
invc <- read.table(file = invc_path)
cmat <- as.matrix(cmat)
invc <- as.matrix(invc)
n <- ncol(cmat)

m <- matrix(1, n, n)
for (i in 1:n) {
  for (j in 1:n){
    if (abs(i - j) < 10 && i != j) {
      m[i, j] <- rho2
    }
    if (j == i) {
      m[i, j] <- rho1
    }
  }
}
res <- glasso(cmat, rho = m, start = "warm", wi.init = invc, w.init = cmat)
write.table(res$wi, file = invg_path)
