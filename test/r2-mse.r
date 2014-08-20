#lasso_ols <- read.table('../results/lasso-ols-v.dat', header=TRUE)
lasso <- read.table('../results/lasso-I.dat', header=TRUE, stringsAsFactors=FALSE)
baart <- read.table('../results/baart-I.dat', header=TRUE, stringsAsFactors=FALSE)

lasso <- lasso[with(lasso, order(Name)),]
baart <- baart[with(baart, order(Name)),]

names <- lasso$Name
galaxies <- c("-LMC-", "-SMC-", "-BLG-")
types <- c("-CEP-", "-RRLYR-", "-T2CEP-", "-ACEP-")
for (galaxy in galaxies) {
  stars_in_galaxy <- grepl(galaxy, lasso$Name)
  
  las <- lasso[stars_in_galaxy,]
  baa <- baart[stars_in_galaxy,]
  
  print(paste(galaxy, nrow(las), mean(las$R.2), mean(baa$R.2), median(las$R.2), median(baa$R.2)))
  
  for (type in types) {
    stars_of_type <- grepl(type, lasso$Name)
    #type_and_galaxy <- intersect(stars_in_galaxy, stars_of_type)
    type_and_galaxy <- stars_in_galaxy & stars_of_type
    
    las <- lasso[type_and_galaxy,]
    baa <- baart[type_and_galaxy,]
    
    print(paste(galaxy, type, nrow(las), mean(las$R.2), mean(baa$R.2), median(las$R.2), median(baa$R.2)))
  }
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