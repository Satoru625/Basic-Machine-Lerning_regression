source("randomforest_regression.R")

# importance
imp <- rf$importance[,1] %>% barplot()
imp

# partial dependence profile
make_pdp_zero <- function(x){
  p <- rf %>% partial(pred.var=x) %>% autoplot(ylab="medv") + 
    geom_smooth(method = "loess") + ylim(c(0, NA))
  return(p)
}

pdps <- map(as.list(names(d)[-14]), ~make_pdp_zero(.))
gridExtra::grid.arrange(grobs=pdps)

make_pdp_zero <- function(x){
  p <- rf %>% partial(pred.var=x) %>% autoplot(ylab="medv") + 
    geom_smooth(method = "loess") + expand_limits(y = 0)
  return(p)
}

pdps_zero <- map(as.list(names(d)[-14]), ~make_pdp(.))
gridExtra::grid.arrange(grobs=pdps_zero)

pdp_3d_lstat_rm <- rf %>% partial(pred.var=c("rm","lstat"),parallel = T) %>%
  plotPartial(levelplot = F,colorkey=F,plot.pdp = F,drape=T,scale=list(arrows=F),plot.margin= unit(c(0, 0, 0, 0), "lines"))

