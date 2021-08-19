#### load packages
library(readr)
library(sp)
library(raster)
library(knitr)
library(rgdal)
library(rgeos)
library(foreach) 
library(doParallel)
library(tidyverse)
library(caret)
library(rattle) # fancyRpartPlot(fit_ctr$finalModel)
library(ggplot2)
library(hrbrthemes)
library(broom)
library(gridExtra)
library(automap)

library(mapview)
library(e1071)

#### set directory
wd <- "~/Documents/test.data/ncrs_back"
setwd(wd)

boundary_dir = paste0(wd, "/boundary")

soil_dir = paste0(wd, "/soilec")

elevation_dir = paste0(wd, "/planting")

weather_dir = paste0(wd, "/weather")

rs_dir = paste0(wd, "/rs_data/ml_")

historical_dir = paste0(wd, "/historical")

#### Data mining 
NCRS_weekly <- read.csv("~/Documents/test.data/ncrs_back/join_data/NCRS_map.csv")
NCRS = na.omit(NCRS_weekly[, -which(colnames(NCRS_weekly) %in% c('X', 'SAVI', 'soil_zones'))])
NCRS = na.omit(NCRS[, which(colnames(NCRS) %in% c("year","x1","x2","yield","soilec_shallow","soilec_deep","elevation","soilom",
                                                  "Rain.7","Rain0","Rain7","Rain14","Rain21","Rain28","Rain35",
                                                  "rad.7","rad0","rad7","rad14","rad21","rad28","rad35",
                                                  "GDD.7","GDD0","GDD7","GDD14","GDD21","GDD28","GDD35"))])


# Parallel processing
cores=detectCores() 
clust_cores <- makeCluster(cores[1]-1) 
registerDoParallel(clust_cores) 

# correlation with the target variable

# cor_target = add_rownames(data.frame(rho = cor(NCRS[,-c(1,2)], NCRS[,2], method = c('pearson'))), "variables")
# ggplot(data=cor_target, aes(x=variables,y=rho)) + geom_bar(position="dodge",stat="identity", fill="steelblue") +
#  coord_flip() + ggtitle("Correlation coefficient (rho) between yield and other variables") +
#  theme_ipsum() + theme(legend.title=element_blank(),legend.position=c(.73,.7), axis.title.y=element_blank(), text=element_text(family="serif",size=20),plot.title=element_text(face="bold",hjust=c(0,0)))

# multicolinearity
 corMatrix = cor(NCRS[,-c(1:4)], method = 'pearson')
 highlyCor = findCorrelation(corMatrix,cutoff = 0.8)
 cor_rm = cbind(NCRS[,-c(1:4)][,-highlyCor], yield=NCRS[,4])

 fit_control <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
 
set.seed(29)
idx_train <- sample(1:nrow(cor_rm), size = nrow(cor_rm)*0.75)

training <- cor_rm[ idx_train, ]
testing  <- cor_rm[ -idx_train, ]

fit_lm =  train(yield ~., data = training, method  = 'lm', trControl = fit_control)
summary(fit_lm)
# caret::varImp(fit_lm)

# fit_lm2 =  train(log(yield) ~., data = training[,-which(colnames(training) %in% var_rm)], method  = 'lm', tuneGrid = expand.grid(intercept = 0)) 
# lmfit_tidy <- tidy(fit_lm2$finalModel)

pred <- predict(fit_lm, testing)
error = postResample(pred = pred, obs = testing$yield)

###### Hold out one year analysis (hooya) for linear model

hooya = cbind(year = NCRS[,1], cor_rm) 

hooya_training = list()
hooya_testing = list()
hooya_fit_lm = list()
hooya_pred = list()
hooya_error = list()

for (i in seq(nlevels(hooya[,1]))) {
  
  hooya_testing[[i]]  = subset(hooya, year == gsub(".shp","",list.files(wd, pattern = "\\.shp$"))[i])
  hooya_training[[i]] = subset(hooya, year != gsub(".shp","",list.files(wd, pattern = "\\.shp$"))[i])
  
  hooya_fit_lm[[i]] =  train(yield ~., data = hooya_training[[i]][,-1], method  = 'lm', trControl = fit_control, tuneGrid = expand.grid(intercept = 0:10)) 
  
  hooya_pred[[i]] <- predict(hooya_fit_lm[[i]], hooya_testing[[i]])
  
  hooya_error[[i]] = postResample(pred = hooya_pred[[i]], obs = hooya_testing[[i]]$yield)
  
}

#############################
###### neural network #######
#############################

set.seed(29)

start_time <- Sys.time() # measure running time
fit_nn <- train(
  
  x = training[,-NCOL(training)],
  y = training$yield/25,
  
  method = "nnet", trControl = fit_control,  
  tuneGrid = expand.grid(size = 1:5, decay = c(0.1, 0.2, 0.3)),
  linout = TRUE, maxit = 500)

end_time <- Sys.time() # measure running time
time_nn = end_time - start_time # 2.374481 hours

fit_nn
plot(fit_nn)

pred_nn <- predict(fit_nn, testing)
error_nn = postResample(pred = pred_nn*25, obs = testing$yield)

hooya_nn = cbind(year = NCRS[,1], cor_rm)

hooya_training2 = list()
hooya_testing2 = list()
hooya_fit_nnet = list()
hooya_pred_nnet = list()
hooya_error_nnet = list()

start_time <- Sys.time() # measure running time
for (i in seq(nlevels(hooya[,1]))) {
  
  hooya_testing2[[i]]  = subset(hooya_nn, year == gsub(".shp","",list.files(wd, pattern = "\\.shp$"))[i])
  hooya_training2[[i]] = subset(hooya_nn, year != gsub(".shp","",list.files(wd, pattern = "\\.shp$"))[i])
  
  set.seed(29)
  
  hooya_fit_nnet[[i]] <- train(
    
    x = hooya_training2[[i]][, -c(1,ncol(hooya_training2[[i]]))],
    y = hooya_training2[[i]]$yield/25,
    
    method = "nnet",trControl = fit_control,
    linout = TRUE, maxit = 500
  )
  
  hooya_pred_nnet[[i]] = predict(hooya_fit_nnet[[i]], hooya_testing2[[i]])
  hooya_error_nnet[[i]] = postResample(pred = hooya_pred_nnet[[i]]*25, 
                                       obs = hooya_testing2[[i]]$yield)
}

end_time <- Sys.time() # measure running time
time_nn2 = end_time - start_time # 13.79714 mins

# library(devtools)
# source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')
# plot.nnet(fit_nn$finalModel)

#####################
###### rpart2 #######
#####################

cor_rm2 = cbind(NCRS[,-c(1:4)], yield=NCRS[,4])

training2 <- cor_rm2[ idx_train, ]
testing2  <- cor_rm2[ -idx_train, ]

fit_ctr <- train(
  
  x = training2[,-NCOL(training2)],
  y = training2$yield,
  
  method = "rpart2",
  trControl = fit_control,
  tuneGrid = expand.grid(maxdepth=1:20)
)

fit_ctr
plot(fit_ctr)

pred_ctr <- predict(fit_ctr, testing2)
error_ctr = postResample(pred = pred_ctr, obs = testing2$yield)

###### Hold out one year analysis (hooya) for rpart
hooya_ctr = cbind(year = NCRS[,1], cor_rm2)

hooya_training3 = list()
hooya_testing3 = list()
hooya_fit_ctr = list()
hooya_pred_ctr = list()
hooya_error_ctr = list()

start_time <- Sys.time()
for (i in seq(nlevels(hooya_ctr[,1]))) {
  
  hooya_testing3[[i]]  = subset(hooya_ctr, year == gsub(".shp","",list.files(wd, pattern = "\\.shp$"))[i])
  hooya_training3[[i]] = subset(hooya_ctr, year != gsub(".shp","",list.files(wd, pattern = "\\.shp$"))[i])
  
  set.seed(29)
  
  hooya_fit_ctr[[i]] <- train(
    
    x = hooya_training3[[i]][, -c(1,ncol(hooya_training3[[i]]))],
    y = hooya_training3[[i]]$yield,
    
    method = "rpart2", tuneGrid = expand.grid(maxdepth = 1:10)
  )
  
  hooya_pred_ctr[[i]] <- predict(hooya_fit_ctr[[i]], hooya_testing3[[i]])
  hooya_error_ctr[[i]] = postResample(pred = hooya_pred_ctr[[i]], obs =hooya_testing3[[i]]$yield)
}

end_time <- Sys.time()
time_ctr2 = end_time - start_time


#####################
####### Cubist ######
#####################

set.seed(29)

start_time <- Sys.time() # measure running time

fit_cubist <- train(
  
  x = training2[,-NCOL(training2)],
  y = training2$yield,
  
  method = "cubist",
  tuneGrid = expand.grid(committees = c(1, 5, 10, 20), neighbors = c(0, 5, 9)),
  trControl = fit_control
)

end_time <- Sys.time() # measure running time
time_cu = end_time - start_time # 3.948066 mins

fit_cubist
plot(fit_cubist)
varImp(fit_cubist)

pred_cubist <- predict(fit_cubist, testing2)
error_cubist = postResample(pred = pred_cubist, obs = testing2$yield)

# predictions <- extractPrediction(list(fit_cubist, fit_lm2), testX = testing[,-NCOL(testing)], testY = log(testing$yield))

# predictions %>% 
#  group_by(model, dataType) %>% 
#  summarise(
#    rmse = RMSE(pred = pred, obs = obs),
#    rsq = R2(pred = pred, obs = obs)
#  )

# plotObsVsPred(predictions)

# exportCubistFiles(fit_cubist$finalModel, neighbors = 1, path = paste0(wd, "/join_data"), prefix = "cubist_model")

###### Hold out one year analysis (hooya) for Cubist
var_incl = c("soilec_shallow", "soilec_deep","soilom",
             "Rain7", 'GDD35',
             "yield")
hooya_cubist = cbind(year = NCRS[,1], cor_rm2[, which(colnames(cor_rm2) %in% var_incl )])

hooya_training4 = list()
hooya_testing4 = list()
hooya_fit_cubist = list()
hooya_pred_cubist = list()
hooya_error_cubist = list()

start_time <- Sys.time() # measure running time
for (i in seq(nlevels(hooya_cubist[,1]))) {
  
  hooya_testing4[[i]]  = subset(hooya_cubist, year == gsub(".shp","",list.files(wd, pattern = "\\.shp$"))[i])
  hooya_training4[[i]] = subset(hooya_cubist, year != gsub(".shp","",list.files(wd, pattern = "\\.shp$"))[i])
  
  set.seed(29)
  
  hooya_fit_cubist[[i]] <- train(
    
    x = hooya_training4[[i]][, -c(1,ncol(hooya_training4[[i]]))],
    y = hooya_training4[[i]]$yield,
    
    method = "cubist",
    trControl = fit_control
  )
  
  hooya_pred_cubist[[i]] <- predict(hooya_fit_cubist[[i]], hooya_testing4[[i]])
  hooya_error_cubist[[i]] = postResample(pred = hooya_pred_cubist[[i]], obs = hooya_testing4[[i]]$yield)
}

end_time <- Sys.time() # measure running time
time_cu2 = end_time - start_time # 9.698936 mins


############################
####### Random forest ######
############################

set.seed(29)

start_time <- Sys.time() # measure running time

fit_rf <- train(
  
  x = training2[,-NCOL(training2)],
  y = training2$yield,
  
  method = "rf",
  trControl = fit_control,
  importance = TRUE
)

end_time <- Sys.time() # measure running time
time_rf = end_time - start_time # 

fit_rf
plot(fit_rf)
varImp(fit_rf)

pred_rf <- predict(fit_rf, testing2)
error_rf = postResample(pred = pred_rf, obs = testing2$yield)

###### Hold out one year analysis (hooya) for Random Forest

hooya_rf = cbind(year = NCRS[,1], cor_rm2)

hooya_training5 = list()
hooya_testing5 = list()
hooya_fit_rf = list()
hooya_pred_rf = list()
hooya_error_rf = list()

start_time <- Sys.time() # measure running time
for (i in seq(nlevels(hooya_rf[,1]))) {
  
  hooya_testing5[[i]]  = subset(hooya_rf, year == gsub(".shp","",list.files(wd, pattern = "\\.shp$"))[i])
  hooya_training5[[i]] = subset(hooya_rf, year != gsub(".shp","",list.files(wd, pattern = "\\.shp$"))[i])
  
  set.seed(29)
  
  hooya_fit_rf[[i]] <- train(
    
    x = hooya_training5[[i]][, -c(1,ncol(hooya_training5[[i]]))],
    y = hooya_training5[[i]]$yield,
    
    method = "rf",
    trControl = fit_control
  )
  
  hooya_pred_rf[[i]] <- predict(hooya_fit_rf[[i]], hooya_testing5[[i]])
  hooya_error_rf[[i]] = postResample(pred = hooya_pred_rf[[i]], obs = hooya_testing5[[i]]$yield)
}

end_time <- Sys.time() # measure running time
time_rf2 = end_time - start_time # 9.698936 mins


######################
####### xgboost ######
######################

# xgb_grid = expand.grid(nrounds = 200, max_depth = c(5, 10, 15), eta = 0.4, gamma = 0, 
#                       colsample_bytree = c(0.5, 0.8), min_child_weight = 1, subsample = 1)

set.seed(29)

start_time <- Sys.time() # measure running time

fit_xg <- train(
  
  x = training2[,-NCOL(training2)],
  y = training2$yield,
  
  method = "xgbTree",
  trControl = fit_control
)

end_time <- Sys.time() # measure running time
time_xg = end_time - start_time # 

fit_xg
plot(fit_xg)

# varImp(fit_xg) 

# xgb_imp <- xgb.importance(feature_names = fit_xg$finalModel$feature_names, model = fit_xg$finalModel)
# xgb.plot.importance(xgb_imp)

pred_xg <- predict(fit_xg, testing2)
error_xg = postResample(pred = pred_xg, obs = testing2$yield)

###### Hold out one year analysis (hooya) for xgboost
var_incl = c("Rain14",
             "soilec_deep",
             "elevation",
             "soilec_shallow",
             "Rain.7",
             "soilom",
             "Rain35",
             "Rain0", 
             "Rain7", "Rain21",
             "Rad21", "Rain28", "GDD0",
             "yield")
hooya_xg = cbind(year = NCRS[,1], cor_rm2[, which(colnames(training2) %in% var_incl )])

hooya_training6 = list()
hooya_testing6 = list()
hooya_fit_xg = list()
hooya_pred_xg = list()
hooya_error_xg = list()

start_time <- Sys.time() # measure running time
for (i in seq(nlevels(hooya_xg[,1]))) {
  
  hooya_testing6[[i]]  = subset(hooya_xg, year == gsub(".shp","",list.files(wd, pattern = "\\.shp$"))[i])
  hooya_training6[[i]] = subset(hooya_xg, year != gsub(".shp","",list.files(wd, pattern = "\\.shp$"))[i])
  
  set.seed(29)
  
  hooya_fit_xg[[i]] <- train(
    
    x = hooya_training6[[i]][, -c(1,ncol(hooya_training6[[i]]))],
    y = hooya_training6[[i]]$yield,
    
    method = "xgbTree",
    trControl = fit_control
  )
  
  hooya_pred_xg[[i]] <- predict(hooya_fit_xg[[i]], hooya_testing6[[i]])
  hooya_error_xg[[i]] = postResample(pred = hooya_pred_xg[[i]], obs = hooya_testing6[[i]]$yield)
}

end_time <- Sys.time() # measure running time
time_xg2 = end_time - start_time # 6.155648 hours


stopCluster(clust_cores) # close connection

# p1 = ggplot(data = data.frame(x = exp(hooya_pred_cubist[[1]]), y = hooya_testing[[1]]$yield)) +
# First layer: the points
#  geom_point(aes(x, y), colour = "royalblue", alpha = 0.5) + coord_cartesian(xlim = c(0, 30), ylim =  c(0, 30)) +
# The 1:1 line
#  geom_abline(slope = 1, intercept = 0, linetype = 2, colour = "darkred") + 
# Labels
#  labs(x = "Predicted", y = "Observed", title = "Prediction of 2014 yield", subtitle = 'RMSE = 6.929163') +
# A nice theme from the hrbrthemes package
#  theme_ipsum()

# grid.arrange(p1, p2, p3, p4, ncol=2, nrow=2)
