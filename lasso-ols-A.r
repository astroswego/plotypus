lasso <- read.table('OGLE-LMC-FU-CEP-I-lasso.dat', header=TRUE, stringsAsFactors=FALSE)
baart <- read.table('OGLE-LMC-FU-CEP-I-ols.dat', header=TRUE, stringsAsFactors=FALSE)

ylim <- c(-0.001, 0.001)

setEPS()
postscript('ogle-i-A.eps', fonts=c("serif"), height=5, width=6.6875)
par(mfrow=c(2,2), mar=c(0, 5, 4, 0), ps=15)
plot(log10(lasso$Period), lasso$A_0 - baart$A_0, pch=3, ylim=ylim,
     xlab="log P", ylab="", xaxt="n", yaxt="n")
abline(h=0)
axis(2)
mtext(side=2, text=expression(Lasso A[0] - Baart A[0]), line=3, cex=0.5)
#"Lasso A₀ - Baart A₀"
par(mar=c(0, 0, 4, 5))
plot(log10(lasso$Period), lasso$A_1 - baart$A_1, pch=3, ylim=ylim,
     xlab="log P", ylab="", xaxt="n", yaxt="n")
abline(h=0)
mtext(side=4, text="Lasso A₁ - Baart A₁", line=3, cex=0.5)

par(mar=c(4, 5, 0, 0))
plot(log10(lasso$Period), lasso$A_2 - baart$A_2, pch=3, ylim=ylim,
     xlab="log P", ylab="", yaxt="n")
abline(h=0)
mtext(side=2, text="Lasso A₂ - Baart A₂", line=3, cex=0.5)
par(mar=c(4, 0, 0, 5))

plot(log10(lasso$Period), lasso$A_3 - baart$A_3, pch=3, ylim=ylim,
     xlab="log P", ylab="", yaxt="n")
abline(h=0)
axis(4)
mtext(side=4, text="Lasso A₃ - Baart A₃", line=3, cex=0.5)

title("Differences in Amplitude Coefficients Between Lasso and Baart", outer=TRUE, line=-3)
dev.off()

# par(mar=c(0, 5, 0, 0))
# plot(log10(lasso$Period), lasso$A_4 - baart$A_4, pch=3, ylim=ylim,
#      xlab="log P", ylab="", xaxt="n", yaxt="n")
# axis(2)
# mtext(side=2, text="Lasso A₄ - Baart A₄", line=3)
# par(mar=c(0, 0, 0, 5))
# plot(log10(lasso$Period), lasso$A_5 - baart$A_5, pch=3, ylim=ylim,
#      xlab="log P", ylab="", xaxt="n", yaxt="n")
# axis(4)
# mtext(side=4, text="Lasso A₅ - Baart A₅", line=3)
# 
# par(mar=c(4, 5, 0, 0))
# plot(log10(lasso$Period), lasso$A_6 - baart$A_6, pch=3, ylim=ylim,
#      xlab="log P", ylab="", yaxt="n")
# axis(2)
# mtext(side=2, text="Lasso A₆ - Baart A₆", line=3)
# par(mar=c(4, 0, 0, 5))
# plot(log10(lasso$Period), lasso$A_7 - baart$A_7, pch=3, ylim=ylim,
#      xlab="log P", ylab="", yaxt="n")
# axis(4)
# mtext(side=4, text="Lasso A₇ - Baart A₇", line=3)

plot(log10(lasso$Period), lasso$A_4 - baart$A_4, pch=3, ylim=ylim,
     xlab="log P", ylab="Lasso A₄ - Baart A₄", xaxt="n")
plot(log10(lasso$Period), lasso$A_5 - baart$A_5, pch=3, ylim=ylim,
     xlab="log P", ylab="Lasso A₅ - Baart A₅", xaxt="n")

plot(log10(lasso$Period), lasso$A_6 - baart$A_6, pch=3, ylim=ylim,
     xlab="log P", ylab="Lasso A₆ - Baart A₆")
plot(log10(lasso$Period), lasso$A_7 - baart$A_7, pch=3, ylim=ylim,
     xlab="log P", ylab="Lasso A₇ - Baart A₇")

