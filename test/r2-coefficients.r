lasso_data <- '../results/lasso-i.dat'
baart_data <- '../results/baart-i.dat'

width <- 6.97
height <- 4.31

# Parse the data
lasso <- read.table(lasso_data, header=TRUE, stringsAsFactors=FALSE)
baart <- read.table(lasso_data, header=TRUE, stringsAsFactors=FALSE)

# Sort the data
lasso <- lasso[with(lasso, order(Name)),]
baart <- baart[with(baart, order(Name)),]

# Remove (likely) bad data
good <- lasso$R.2 > 0 | baart$R.2 > 0
lasso <- lasso[good,]
baart <- baart[good,]

# Make a table of results 
names <- lasso$Name
galaxies <- c("-LMC-", "-SMC-", "-BLG-")
types <- c("-CEP-", "-T2CEP-", "-ACEP-", "-RRLYR-")

table_row <- function(galaxy, type, las, baa) {
  N <- las$Inliers+las$Outliers
  stopifnot(N>0)
  lasAVG <- median(las$R.2)
  lasMAD <- mad(las$R.2)
  baaAVG <- median(baa$R.2)
  baaMAD <- mad(baa$R.2)
  p.value <- wilcox.test(las$R.2, baa$R.2, paired=TRUE, alternative="greater")$p.value
  lasso_wins <- p.value < .0001#lasAVG - lasMAD > baaAVG + baaMAD
  baart_wins <- p.value > .9999#baaAVG - baaMAD > lasAVG + lasMAD
  paste(gsub("-", "", galaxy), '&',
        gsub("-", "", type), '&',
        nrow(las), '&',
        format(mean(N), digits=4, nsmall=1),
        paste("(", format(sd(N), digits=4, nsmall=1), ') &', sep=""),
        ifelse(lasso_wins, "\\textbf{", ""),
        format(lasAVG, digits=4, nsmall=4,
               scientific=ifelse(abs(lasAVG)>10, TRUE, FALSE)),
        paste("(", format(round(lasMAD, 4), digits=4, nsmall=4,
                         scientific=ifelse(abs(lasMAD)>10, TRUE, FALSE)),
        ")", ifelse(lasso_wins, "}", ""), " &", sep=""),
        ifelse(baart_wins, "\\textbf{", ""),
        format(baaAVG, digits=4, nsmall=4,
               scientific=ifelse(abs(baaAVG)>10, TRUE, FALSE)),
        paste("(", format(round(baaMAD, 4), digits=4, nsmall=4,
                         scientific=ifelse(abs(baaMAD)>10, TRUE, FALSE)),
        ")", ifelse(baart_wins, "}", ""), " &", sep=""),
        "$p", ifelse(p.value < .0001, "< .0001$",
                    paste("=", format(p.value, digits=4, nsmall=4), "$")), "\\\\")
}

out <- table_row('(all)', '(all)', lasso, baart)
for (type in types) {
  stars_of_type <- grepl(type, lasso$Name)
  las <- lasso[stars_of_type,]
  baa <- baart[stars_of_type,]
  if (nrow(las) > 0)
    out <- paste(out, table_row('(all)', type, las, baa), sep='\n')
}
for (galaxy in galaxies) {
  stars_in_galaxy <- grepl(galaxy, lasso$Name)
  las <- lasso[stars_in_galaxy,]
  baa <- baart[stars_in_galaxy,]
  if (nrow(las) > 0)
    out <- paste(out, table_row(galaxy, '(all)', las, baa), sep='\n')
  for (type in types) {
    stars_of_type <- grepl(type, lasso$Name)
    type_and_galaxy <- stars_in_galaxy & stars_of_type
    las <- lasso[type_and_galaxy,]
    baa <- baart[type_and_galaxy,]
    if (nrow(las) > 0)
      out <- paste(out, table_row(galaxy, type, las, baa), sep='\n')
  }
}
cat(out)#of(the)bag

# Plot the differences in amplitude components
good_models <- lasso$R.2 > 0.99 & baart$R.2 > 0.99
las <- lasso[good_models,]
baa <- baart[good_models,]

ylim <- c(-0.003, 0.003)
color <- function(las) {
  ifelse(grepl("-CEP-", las$Name), "darkred",
         ifelse(grepl("-T2CEP-", las$Name), "darkblue",
                      ifelse(grepl("-ACEP-", las$Name), "darkgreen", 
                             "gray12")))
}
pch <- function(las) {
  ifelse(grepl("-CEP-", las$Name), 4,
         ifelse(grepl("-T2CEP-", las$Name), 2,
                ifelse(grepl("-ACEP-", las$Name), 1, 
                       3)))
}
make_legend <- function() {
  legend("topright", c("RRLYR", "CEP", "T2CEP", "ACEP"), pch=c(3, 4, 2, 1),
         col=c("gray12", "darkred", "darkblue", "darkgreen"), bg="white")
}

setEPS()
postscript('../results/ogle-i-A.eps', fonts=c("serif"), width=width, height=height)
par(mfrow=c(2,2), ps=10, family=c("serif"))

par(mar=c(1.5, 3, 1.75, 0.5))
plot(log10(las$Period), las$A_0 - baa$A_0,
     pch=pch(las), col=color(las), xlab="", ylab="", xaxt="n", yaxt="n", ylim=ylim)
abline(h=0)
axis(2)
mtext(side=2, text=expression('Lasso A'[0]*' - Baart A'[0]), line=2, cex=1)

par(mar=c(1.5, 0.5, 1.75, 3))
plot(log10(las$Period), las$A_1 - baa$A_1, pch=pch(las), col=color(las),
     xlab="", ylab="", xaxt="n", yaxt="n", ylim=ylim)
abline(h=0)
axis(4)
mtext(side=4, text=expression('Lasso A'[1]*' - Baart A'[1]), line=2, cex=1)

par(mar=c(3.25, 3, 0, 0.5))
plot(log10(las$Period), las$A_2 - baa$A_2, pch=pch(las), col=color(las),
     xlab="", ylab="", yaxt="n", ylim=ylim)
abline(h=0)
axis(2)
mtext(side=2, text=expression('Lasso A'[2]*' - Baart A'[2]), line=2, cex=1)
mtext(side=1, text="log P", line=2.25, cex=1)

par(mar=c(3.25, 0.5, 0, 3))
plot(log10(las$Period), las$A_3 - baa$A_3, pch=pch(las), col=color(las),
     xlab="", ylab="", yaxt="n", ylim=ylim)
make_legend()
abline(h=0)
axis(4)
mtext(side=4, text=expression('Lasso A'[3]*' - Baart A'[3]), line=2, cex=1)
mtext(side=1, text="log P", line=2.25, cex=1)

par(ps=13)
title("Differences in Amplitude Coefficients Between Lasso and Baart",
      outer=TRUE, line=-0.85, font.main=1)
dev.off()

# Find stars with missing amplitude components
amplitudes <- function(stars) stars[grepl("^A_\\d+$", names(stars))]
n_components <- function(amps) sum(ifelse(amps, 1, 0))
max_degree <- function(amps) ifelse(amps[length(amps)], length(amps) - 1,
                                    max(which(diff(ifelse(amps, 1, 0)) == -1)) - 1)
lasso_degrees <- apply(amplitudes(lasso), 1, max_degree)
baart_degrees <- apply(amplitudes(baart), 1, max_degree)
has_absents <- apply(amplitudes(lasso) != 0, 1, function(x) {1 %in% diff(x)})
absents <- lasso[has_absents
                 & lasso$R.2 > baart$R.2
                 & lasso$R.2 > 0.99
                 & lasso$Coverage > 0.95
                 & lasso_degrees <= baart_degrees
                 & lasso_degrees <= 15,]
for (i in 1:nrow(absents)) {
  amps <- amplitudes(absents[i,])
  degree <- max_degree(amps)
  missing <- which(amps[1:(1+degree)] == 0) - 1
  cat(paste(absents[i,]$Name, '&',
            absents[i,]$Period, '&',
            absents[i,]$Inliers + absents[i,]$Outliers, '&',
            format(absents[i,]$A_0, digits=4, nsmall=4), '&',
            max_degree(amplitudes(baart[baart$Name %in% absents[i,]$Name,])), '&',
            degree, '&', paste(missing, collapse=", "), '\\\\'))
  writeLines("")
}

# Plot R^2 difference as a function of coverage and sample size
setEPS()
postscript('../results/ogle-i-difference.eps', fonts=c("serif"), height=height, width=width)

par(mfrow=c(1,2), ps=10, family=c("serif"))
nonzero <- lasso$R.2 > 0 & baart$R.2 > 0
nonzero <- nonzero & ifelse(-0.01 < lasso$R.2 - baart$R.2 & lasso$R.2 - baart$R.2 < .01,
                            runif(length(nonzero)) > .95, 1)
difference <- lasso[nonzero,]$R.2 - baart[nonzero,]$R.2
N <- lasso[nonzero,]$Inliers + lasso[nonzero,]$Outliers
coverage <- lasso[nonzero,]$Coverage

par(mar=c(3.25, 3, 1.75, 0))
plot(log10(N), difference,
     pch=pch(las), col=color(las),
     ylim=c(-0.1, 1), yaxs="i",
     xlab="",
     ylab="")
mtext(side=1, text=expression('log'[10]*' Observations'), line=2.25, cex=1)
mtext(side=2, text=expression('Lasso R'^2*' - Baart R'^2), line=2, cex=1)
make_legend()
abline(h=0)

par(mar=c(3.25, 1, 1.75, 2))
plot(coverage, difference,
     pch=pch(las), col=color(las),
     xlab="", ylab="", yaxt="n",
     xlim=c(0.2, 1), ylim=c(-0.1, 1), 
     xaxs="i", yaxs="i")
mtext(side=1, text="Phase Coverage", line=2.25, cex=1)
axis(4)
abline(h=0)

par(ps=11)
title("Differences in Determination Coefficients by Sample Size and Coverage",
      outer=TRUE, line=-0.85, font.main=1)
dev.off()

# Make box plots of MSE and R^2
# setEPS()
# postscript('ogle-v-r2.eps', fonts=c("serif"), height=5, width=6.6875)
# #postscript('ogle-v-r2.eps', fonts=c("serif"), height=2, width=2.675)
# #par(family="serif", ps=7, mar=c(1.95,2.55,0.85,0.3))
# par(family="serif", ps=15, mar=c(2,4,1.5,0.1))
# boxplot(lasso$R.2, baart$R.2, ylim=c(0.3, 1), pch=3, ylab="R²", las=1, outline=FALSE,
#         names=c("Cross-Validated Lasso", "Baart's Least Squares"), ann=FALSE,
#         pars=list(boxwex=0.95, boxlwd=0.75, medlwd=0.75, whisklwd=0.75, staplelwd=0.75))
# box(lwd=0.3)
# #mtext(side=2, text="R²", line=2)
# #par(ps=15)
# #par(ps=8)
# title(main="Determination Coefficients for OGLE V-Band Photometry", font.main=1)
# dev.off()
# 
# setEPS()
# postscript('ogle-v-mse.eps', fonts=c("serif"), height=5, width=6.6875)
# par(family="serif", ps=15, mar=c(2,4,1.6,0.1))
# boxplot(abs(lassoMSE), abs(olsMSE), ylim=c(0, 0.016), pch=3, ylab="MSE", las=1, outline=FALSE,
#         names=c("Cross-Validated Lasso", "Baart's Least Squares"), ann=FALSE,
#         pars=list(boxwex=0.95, boxlwd=0.75, medlwd=0.75, whisklwd=0.75, staplelwd=0.75))
# box(lwd=0.3)
# title(main="Mean Squared Errors for OGLE V-Band Photometry", font.main=1)
# dev.off()
