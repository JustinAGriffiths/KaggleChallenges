library('devtools')
library('roxygen2')
library('getopt')


spec = matrix(c(
    'package', 'p', 1, "character"
    ), byrow=TRUE, ncol=4);
opt = getopt(spec);

pack<-opt$package

setwd(pack)
document()
setwd('..')
install(pack)
