library(MASS)
library(GGally)
library(patchwork)
d <- Boston

# correlation matrix
g <- ggpairs(d)

# boxplots
gs <- map2(d,as.list(names(d)), function(x,y) ggplot(d) + geom_boxplot(aes(x)) + 
             coord_flip() + labs(x="",y=y) + theme(axis.text.x=element_blank(),
                                                axis.ticks.x=element_blank() ))
gridExtra::grid.arrange(grobs=gs)
