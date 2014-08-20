library(plotrix)

#lasso_ols <- read.table('../results/lasso-ols-v.dat', header=TRUE)
lasso <- read.table('../results/lasso-I.dat', header=TRUE, stringsAsFactors=FALSE)
baart <- read.table('../results/baart-I.dat', header=TRUE, stringsAsFactors=FALSE)

lasso <- lasso[with(lasso, order(Name)),]
baart <- baart[with(baart, order(Name)),]

table_row <- function(galaxy, type, las, baa) {
  N <- las$Inliers+las$Outliers
  lasAVG <- mean(las$R.2)
  lasSEM <- std.error(las$R.2)
  baaAVG <- mean(baa$R.2)
  baaSEM <- std.error(baa$R.2)
  lasso_wins <- lasAVG - lasSEM > baaAVG + baaSEM
  baart_wins <- baaAVG - baaSEM > lasAVG + lasSEM
  paste(gsub("-", "", galaxy), '&',
        gsub("-", "", type), '&',
        nrow(las), '&',
        format(mean(N), digits=4), "$\\pm$", format(sd(N), digits=4), '&',
        ifelse(lasso_wins, "\\textbf{", ""),
        format(lasAVG, digits=4, scientific=ifelse(abs(lasAVG)>10, TRUE, FALSE)),
        "$\\pm$", format(lasSEM, digits=4, scientific=ifelse(abs(lasSEM)>10, TRUE, FALSE)),
        ifelse(lasso_wins, "}", ""), '&',
        ifelse(baart_wins, "\\textbf{", ""),
        format(baaAVG, digits=4, scientific=ifelse(abs(baaAVG)>10, TRUE, FALSE)),
        "$\\pm$", format(baaSEM, digits=4, scientific=ifelse(abs(baaSEM)>10, TRUE, FALSE)),
        ifelse(baart_wins, "}", ""), "\\\\")
}

names <- lasso$Name
galaxies <- c("-LMC-", "-SMC-", "-BLG-")
types <- c("-CEP-", "-RRLYR-", "-T2CEP-", "-ACEP-")
for (galaxy in galaxies) {
  stars_in_galaxy <- grepl(galaxy, lasso$Name)
  las <- lasso[stars_in_galaxy,]
  baa <- baart[stars_in_galaxy,]
  cat(table_row(galaxy, '(all)', las, baa))
  writeLines("")
  
  for (type in types) {
    stars_of_type <- grepl(type, lasso$Name)
    type_and_galaxy <- stars_in_galaxy & stars_of_type
    las <- lasso[type_and_galaxy,]
    baa <- baart[type_and_galaxy,]
    cat(table_row(galaxy, type, las, baa))
    writeLines("")
  }
}
for (type in types) {
  stars_of_type <- grepl(type, lasso$Name)
  las <- lasso[stars_of_type,]
  baa <- baart[stars_of_type,]
  cat(table_row('(all)', type, las, baa))
  writeLines("")
}

setEPS()
postscript('ogle-v-r2.eps', fonts=c("serif"), height=5, width=6.6875)
#postscript('ogle-v-r2.eps', fonts=c("serif"), height=2, width=2.675)
#par(family="serif", ps=7, mar=c(1.95,2.55,0.85,0.3))
par(family="serif", ps=15, mar=c(2,4,1.5,0.1))
boxplot(lasso$R.2, baart$R.2, ylim=c(0.3, 1), pch=3, ylab="R²", las=1, outline=FALSE,
        names=c("Cross-Validated Lasso", "Baart's Least Squares"), ann=FALSE,
        pars=list(boxwex=0.95, boxlwd=0.75, medlwd=0.75, whisklwd=0.75, staplelwd=0.75))
box(lwd=0.3)
#mtext(side=2, text="R²", line=2)
#par(ps=15)
#par(ps=8)
title(main="Determination Coefficients for OGLE V-Band Photometry", font.main=1)
dev.off()

setEPS()
postscript('ogle-v-mse.eps', fonts=c("serif"), height=5, width=6.6875)
par(family="serif", ps=15, mar=c(2,4,1.6,0.1))
boxplot(abs(lassoMSE), abs(olsMSE), ylim=c(0, 0.016), pch=3, ylab="MSE", las=1, outline=FALSE,
        names=c("Cross-Validated Lasso", "Baart's Least Squares"), ann=FALSE,
        pars=list(boxwex=0.95, boxlwd=0.75, medlwd=0.75, whisklwd=0.75, staplelwd=0.75))
box(lwd=0.3)
title(main="Mean Squared Errors for OGLE V-Band Photometry", font.main=1)
dev.off()
