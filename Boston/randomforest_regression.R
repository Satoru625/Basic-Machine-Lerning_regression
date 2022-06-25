library(MASS)
library(tidyverse)
library(randomForest)
library(pdp)

d <- Boston
idx <- sample(x=1:nrow(d),size = as.integer((nrow(d)*0.7)),replace = F)
d_train <- d[idx,]
d_test <- d[-idx,]

# train and fit
tune <- tuneRF(x=d_train[,-14],y=d_train[,14],plot = T,doBest = T)
rf <- randomForest(medv~.,data = d_train, mtry = tune$mtry)
fit.rf <- predict(rf,d_train)
rf_act <- lm(fit.rf~d$medv[idx])
rf_act.sum <- summary(rf_act)
R2.rf <- paste("Rsquare",round(rf_act.sum$r.squared,2),sep = ":")
rmse.rf <- paste("RMSE",round(sqrt(mean((d$medv[idx]-fit.rf)^2)),2),sep = ":")
#result plot
plot(d$medv[idx],fit.rf)
abline(rf_act,col="red")
text(x=40,y=0,R2.rf,col="blue")
text(x=40,y=10,rmse.rf,col="blue")

# test
fit.test <- predict(rf,d_test)
test_act <- lm(fit.test~d$medv[-idx])
test_act.sum <- summary(test_act)
R2.test <- paste("Rsquare",round(test_act.sum$r.squared,2),sep = ":")
rmse.test <- paste("RMSE",round(sqrt(mean((d$medv[-idx]-fit.test)^2)),2),sep = ":")
#result plot
plot(d$medv[-idx],fit.test)
abline(test_act,col="red")
text(x=40,y=0,R2.test,col="blue")
text(x=40,y=10,rmse.test,col="blue")


