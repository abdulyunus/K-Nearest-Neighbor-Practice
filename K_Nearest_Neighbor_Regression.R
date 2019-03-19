# K Nearest Neighbor Regression in R

# KNN can be used for both classification and regression predictive problems.
# However, it is more widely used in classification problems in the industry. To evaluate any technique we generally look at 3 important aspects:

# 1. Ease to interpret output
# 2. Calculation time
# 3. Predictive Power

# KNN is a non-parametric, lazy learning algorithm. 
# Its purpose is to use a database in which the data points are separated into several classes to predict the classification of a new sample point.
# When we say a technique is non-parametric , it means that it does not make any assumptions on the underlying data distribution. 
# In other words, the model structure is determined from the data. 
# If you think about it, it's pretty useful, because in the "real world", most of the data does not obey the typical theoretical assumptions made (as in linear regression models, for example). 
# Therefore, KNN could and probably should be one of the first choices for a classification study when there is little or no prior knowledge about the distribution data.
# KNN Algorithm is based on feature similarity: How closely out-of-sample features resemble our training set determines how we classify a given data point.

# KNN use Ecluidean distence or Manhattan distance to identify the nearest method. Ecluidean method is the most common one.

# K nearest classification

# Installed the required packages
gc()
rm(list = ls(all = TRUE))

packages<-function(x){
  x<-as.character(match.call()[[2]])
  if (!require(x,character.only=TRUE)){
    install.packages(pkgs=x,repos="http://cran.r-project.org")
    require(x,character.only=TRUE)
  }
}

packages(caret)
packages(caTools)
packages(pROC)
packages(mlbench)

# Here we are using Boston Housing data from package 'mlbench'

data("BostonHousing")

df = BostonHousing

str(df)

# Data Partitioning
set.seed(123)
id = sample.split(Y = df$medv, SplitRatio = 0.75)
train_df = subset(df, id == "TRUE")
test_df = subset(df, id == "FALSE")


# Before we made K nearest neighbor model, lets specify train COntrol, this will be used in the model
trControl = trainControl(method = "repeatedcv",
                         number = 10,
                         repeats = 3)

set.seed(123)
fit = train(medv ~., data = train_df,
            method = 'knn',
            tuneLength = 20,
            trControl = trControl,
            preProcess = c('center', 'scale'),
            tuneGrid = expand.grid(k = 1:70))

# The tuneLength parameter tells the algorithm to try different default values for the main parameter
# In this case we used 20 default values

# The tuneGrid parameter lets us decide which values the main parameter will take
# While tuneLength only limit the number of default parameters to use.

fit
plot(fit)

varImp(fit)

# Lets predict the result using the test data set and then form a confusion matrix

pred = predict(object = fit, newdata = test_df)

# Since we are dealing with numerical variable prediction (Regression problem) therefore, now we find the RMSE for accuracy. We will not do the confusion matrix here

RMSE(pred = pred, test_df$medv)


# Lets plot the predicted values

plot(pred, test_df$medv)


# We can also specify the accuracy matric in the model itself. Lets use R2, instead of RMSE(default)

set.seed(123)
fit = train(medv ~., data = train_df,
            method = 'knn',
            tuneLength = 20,
            metric = 'Rsquared',
            trControl = trControl,
            preProcess = c('center', 'scale'),
            tuneGrid = expand.grid(k = 1:70))

fit
plot(fit)

varImp(fit)

# Lets predict the result using the test data set and then form a confusion matrix

pred = predict(object = fit, newdata = test_df)

# Since we are dealing with numerical variable prediction (Regression problem) therefore, now we find the RMSE for accuracy. We will not do the confusion matrix here

RMSE(pred = pred, test_df$medv)


# Lets plot the predicted values

plot(pred, test_df$medv)
