# Writes data/bei.csv: locations (in metres) of 3,604 Beilschmiedia pendula trees
# on Barro Colorado Island (Hubbell and Foster, 1983), from the R package
# spatstat.data. Run from the repository root:
#
#   Rscript data/get_bei.R
#
# Requires install.packages("spatstat.data").
library(spatstat.data)
data(bei)
write.csv(data.frame(x1 = bei$x, x2 = bei$y), "data/bei.csv", row.names = FALSE, quote = FALSE)
