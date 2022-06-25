library(MASS)
library(tidyverse)
library(glmnet)
library(glmnetUtils)

d <- Boston
medv <- as.matrix(d[,14])
var <- as.matrix(d[,-14])
lasso.reg.cv <- cv.glmnet(x=var, y=medv,
                           nfolds = 10,
                           alpha = 1, 
                           standardize = TRUE)
bestlambda <- lasso.reg.cv$lambda.min
coef(lasso.reg.cv, s = bestlambda)
# 14 x 1 sparse Matrix of class "dgCMatrix"
# s1
# (Intercept)  34.248405456
# crim         -0.097424931
# zn            0.041034124
# indus         .          
# chas          2.681496880
# nox         -16.205420456
# rm            3.872832939
# age           .          
# dis          -1.386908239
# rad           0.248393350
# tax          -0.009648008
# ptratio      -0.928430214
# black         0.009000473
# lstat        -0.522491506
fit.model <- predict(lasso.reg.cv, newx = var, s = bestlambda, alpha = 1)
las <- lm(fit.model~d$medv)
las.sum <- summary(las)
R2.las <- paste("Rsquare",round(las.sum$r.squared,2),sep = ":")
rmse.las <- paste("RMSE",round(sqrt(mean((medv-fit.model)^2)),2),sep = ":")
#result plot
plot(d$medv,fit.model)
abline(las,col="red")
text(x=40,y=0,R2.las,col="blue")
text(x=40,y=10,rmse.las,col="blue")
