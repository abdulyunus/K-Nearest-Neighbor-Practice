# K Nearest Neighbor in R

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

# Set the working directory
setwd("C:/Users/Abdul_Yunus/Desktop/Yunus_Personal/Learning/KNN")

# import the diabetics.csv file

df = read.csv("diabetes.csv", header = T)

head(df)
str(df)

# Outcome is our target (Predicting) variable  and it is numerical, lets convert it first into factor
df$Outcome = as.factor(ifelse(df$Outcome == 1, "YES", "NO"))

# Data Partitioning
set.seed(123)
id = sample.split(Y = df$Outcome, SplitRatio = 0.75)
train_df = subset(df, id == "TRUE")
test_df = subset(df, id == "FALSE")

# Before we made K nearest neighbor model, lets specify train COntrol, this will be used in the model
trControl = trainControl(method = "repeatedcv",
                         number = 10,
                         repeats = 3)

# Lets set the seet to get the repeatability of the outcome

set.seed(123)
fit = train(Outcome ~., data = train_df,
            method = 'knn',
            tuneLength = 20,
            trControl = trControl,
            preProcess = c('center', 'scale'))

# Lets check how the model perform
fit

# We can plot the model to check which K value we can consider for highest accuracy.
plot(fit)

# Lets check the variable importance
varImp(fit)

# Lets predict the result using the test data set and then form a confusion matrix

pred = predict(object = fit, newdata = test_df)

confusionMatrix(pred, test_df$Outcome)

# We can also change the method to check accuracy, we can also use ROC curve for accuracy. For this we need to make some changes

trControl = trainControl(method = "repeatedcv",
                         number = 10,
                         repeats = 3,
                         classProbs = TRUE,
                         summaryFunction = twoClassSummary)
# Here we have added 'ClassProbs' and 'summaryFunction' into trainControl, which is required to have method as ROC
# Lets set the seet to get the repeatability of the outcome

set.seed(123)
fit = train(Outcome ~., data = train_df,
            method = 'knn',
            tuneLength = 20,              # The tuneLength parameter tells the algorithm to try different default values for the main parameter
            trControl = trControl,
            preProcess = c('center', 'scale'),
            metric = 'ROC',
            tuneGrid = expand.grid(k = 1:60))

# Lets check how the model perform
fit

# Here best ROP value we are getting at k = 44
# We can plot the model to check which K value we can consider for highest accuracy.
plot(fit)

# Lets check the variable importance
varImp(fit)

# Lets predict the result using the test data set and then form a confusion matrix

pred = predict(object = fit, newdata = test_df)

confusionMatrix(pred, test_df$Outcome)


# From confusion metrix, we can see that accuracy has slightly improved

attributes(fit)
fit$finalModel
