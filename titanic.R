
# Kaggle Starter Competition 
# Titanic prediction
# Spencer Ebert

# The purpose of this file is to perform classification analysis using logistic
#   regression, random forests, and boosting then compare which
#   method is the most effective.

# Load required libraries
library(tidyverse)
library(randomForest)
library(xgboost)

# Read in the test and train data
# I will split into train and test for the train data so I name them differently here
titanic.pred <- read.csv("test.csv")
titanic <- read.csv("train.csv")

# Read in the example file of a submission
gender_submission <- read.csv("gender_submission.csv")


###############
##### EDA #####
head(titanic)
# Meaning for each column
#   PassengerId     Unique ID of passenger
#   Survived        (No=0, Yes=1)
#   Pclass          Socio Economic status of passenger
#   Name            Name of passenger
#   Sex             male or female
#   Age             Age of passenger (there are some NA values)
#   SibSp           Number of siblings / spouses aboard
#   Parch           Number of children / parents aboard
#   Ticket          Ticket Number
#   Fare            Passenger Fare
#   Cabin           Cabin Number (Most values are empty)
#   Embarked        Port of Embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)


# Graphs for each of these variables
# Effect of Sex
titanic %>% 
  group_by(Sex) %>%
  summarise(Prop_Surv = sum(Survived)/n()) %>%
  ggplot(aes(x=Sex,y=Prop_Surv)) +
  geom_bar(stat="identity") +
  ylim(0,1)
# Women survived more than men

# Effect of Class
titanic %>% 
  group_by(Pclass) %>%
  summarise(Prop_Surv = sum(Survived)/n()) %>%
  ggplot(aes(x=Pclass,y=Prop_Surv)) +
  geom_bar(stat="identity") +
  ylim(0,1)
# Looks like higher class survived

# Effect of Siblings
titanic %>%
  group_by(SibSp) %>%
  summarise(Prop_Surv = sum(Survived)/n(), n_people = n()) %>%
  ggplot(aes(x=SibSp,y=Prop_Surv)) +
  geom_bar(stat="identity") +
  ylim(0,1)
# Looks like 1 or 2 siblings/spouses survived. Could be confounded with other variables

# Effect of Parent/Children
titanic %>%
  group_by(Parch) %>%
  summarise(Prop_Surv = sum(Survived)/n(), n_people = n()) %>%
  ggplot(aes(x=Parch,y=Prop_Surv)) +
  geom_bar(stat="identity") +
  ylim(0,1)
# There doesn't seem to be much of a relationship here

# Effect of Fare
titanic %>%
  ggplot(aes(x=Fare,y=Survived,color=Sex)) +
  geom_point(size=3,alpha=0.5) +
  geom_smooth(color="black",se=FALSE)
# It looks like people with higher fares were more likely to survive. Particularly men with fare
#   above 500 survived (2 men)
titanic %>% 
  select(PassengerId,Sex,Fare) %>%
  filter(Fare>500)
###############

###############
# Check to see what the target prediction data looks like (missing values, classes, etc.) #####
titanic.pred$GivenCabin <- as.factor(as.numeric(titanic.pred$Cabin != ""))
summary(titanic.pred)
# Fare has one NA value
titanic.pred[which(is.na(titanic.pred$Fare)),]
# I fill in the value with the average fare (since it is only one observation and this person is going to be marked
#   as not surviving anyways)
titanic.pred[which(is.na(titanic.pred$Fare)),]$Fare <- mean(titanic.pred$Fare[-which(is.na(titanic.pred$Fare))])
# Change class to a factor
titanic.pred$Pclass <- as.factor(titanic.pred$Pclass)
###############

#### The analysis ####


# Since there are missing values for age I do logistic regression on whether age is missing or not to see if missingness is at random
titanic$MissingAge <- as.factor(as.numeric(is.na(titanic$Age)))
titanic$GivenCabin <- as.factor(as.numeric(titanic$Cabin != ""))

age.logit <- glm(MissingAge~Pclass+Sex+Survived+SibSp+Parch+Fare+Embarked+GivenCabin,data=titanic,family="binomial")
summary(age.logit)
table(titanic$MissingAge,titanic$Pclass)
# It seems as though missingness in age is lower for second class tickets and higher for number of siblings

# Embarked has 2 missing values so I fill those values with S the most common one 
titanic$Embarked[which(titanic$Embarked=="")] <- "S"
titanic$Embarked <- factor(titanic$Embarked)

# I am going to use 2 methods to deal with the missing values in Age
# 1: Use linear regression to fill in the age. Since I'm looking at a point prediction, I am not concerned about
#   doing multiple imputation because I'm not trying to create confidence intervals or t-tests to know the 
#   significance of the particular variable
# 2: Group Age into multiple groups Kid(0-16), Teenage-YA (17-21), Young Adult (22-30), Middle Aged (31-55),
#   Older (56+), Missing Age 

# Method 1
lm.missing <- lm(data=titanic,Age~Pclass+Sex+SibSp+Parch+Fare+Embarked+GivenCabin) # lm here excludes rows where Age is missing
lm.missing$fitted.values
pred.lm.missing <- predict(lm.missing,newdata=titanic[is.na(titanic$Age),])
titanic$Age1 <- titanic$Age
titanic$Age1[is.na(titanic$Age1)] <- pred.lm.missing

# Method 2
titanic$Age2 <- titanic$Age
titanic$Age2[is.na(titanic$Age)] <- 'MissingAge'
titanic$Age2[titanic$Age<=16 & !is.na(titanic$Age)] <- 'Kid'
titanic$Age2[titanic$Age>16 & titanic$Age<=21 & !is.na(titanic$Age)] <- 'Teenage'
titanic$Age2[titanic$Age>21 & titanic$Age<=30 & !is.na(titanic$Age)] <- 'YA'
titanic$Age2[titanic$Age>30 & titanic$Age<=55 & !is.na(titanic$Age)] <- 'Middle'
titanic$Age2[titanic$Age>55 & !is.na(titanic$Age)] <- 'Older'


# Now that I have the two methods for missing age I am going to use logistic regression, random forests, and xgboost
#   to predict Survival and use the best model. To determine the effectiveness of each method I utilize leave one
#   out cross validation

# Logistic regression
titanic$Pclass <- as.factor(titanic$Pclass)
titanic$GivenCabin <- as.factor(as.numeric(titanic$Cabin != ""))
table(titanic$Pclass,titanic$GivenCabin)
# Most of the people with a cabin number are first class so there is some colinearity there

logit.fit <- glm(Survived~Pclass+Sex+Age1+SibSp+Parch+Fare+Embarked+GivenCabin,data=titanic,family="binomial")
summary(logit.fit)
predict(logit.fit,newdata = titanic[1,], type="response")

# logit leave one out cross validation function
logit_crossVal <- function(dat, missingness=1, cutoff=0.5) {
  n <- nrow(dat)
  the_vals <- numeric(n)
  for (i in 1:n) {
    dat_leave_out <- dat[-i,]
    if (missingness==1) {
      # Fit model without age
      cv_logit_fit <- glm(Survived~Pclass+Sex+Age1+SibSp+Parch+Fare+Embarked+GivenCabin,data=dat_leave_out,family="binomial")
    } else {
      # fit model with age
      cv_logit_fit <- glm(Survived~Pclass+Sex+Age2+SibSp+Parch+Fare+Embarked+GivenCabin,data=dat_leave_out,family="binomial")
    }
    the_vals[i] <- predict(cv_logit_fit, newdata=dat[i,], type="response")
  }
  list(correct_rate=mean((the_vals>cutoff)==dat$Survived),prediction_percentage=the_vals)
}

logistic_leave_one_out1 <- logit_crossVal(titanic,missingness=1) #81% correct prediction rate
logistic_leave_one_out2 <- logit_crossVal(titanic,missingness=2) #79% correct prediction rate
# Grid search for the best cutoff
cutoff_vals <- seq(0,1,by=0.01)
choose_cutoff <- sapply(cutoff_vals, function(x) mean( (logistic_leave_one_out1$prediction_percentage>x)==titanic$Survived ))
cutoff_vals[which.max(choose_cutoff)]
# The best cutoff is at 0.55. This gives a correct prediction performance of 82%. Looking at method  for missingness


# Random forests

titanic$Survived <- as.factor(titanic$Survived)
rf1 <- randomForest(Survived~Pclass+Sex+Age1+SibSp+Parch+Fare+Embarked+GivenCabin,data=titanic,ntree=500)
predict(rf1,newdata=titanic[17,])
mean(predict(rf1,newdata=titanic)==titanic$Survived)

# leave one out cross validation to choose optimal parameters in the random forest
rf_crossVal <- function(dat,ntree,missingness=1) {
  n <- nrow(dat)
  pb <- txtProgressBar(min=0,max=n,style=3)
  the_vals <- numeric(n)
  for (i in 1:n) {
    dat_leave_out <- dat[-i,]
    if (missingness==1) {
      # Fit model without age
      rf_fit <- randomForest(Survived~Pclass+Sex+Age1+SibSp+Parch+Fare+Embarked+GivenCabin,data=dat_leave_out,ntree=ntree)
    } else {
      # fit model with age
      rf_fit <- randomForest(Survived~Pclass+Sex+Age2+SibSp+Parch+Fare+Embarked+GivenCabin,data=dat_leave_out,ntree=ntree,na.action = na.omit)
    }
    the_vals[i] <- predict(rf_fit, newdata=dat[i,])
    setTxtProgressBar(pb,i)
  }
  mean(the_vals-1==dat$Survived) # minus 1 here because the prediction is in 1 and 2
}
rf_crossVal(titanic,ntree=100,missingness=1) # 82% Correct prediction rate


# Test different values for ntree to see which performs the best
ntree.test <- 2:10 * 100
correct_rates <- sapply(ntree.test, FUN=rf_crossVal, dat=titanic, missingness=1)
# 300 gave the best prediction with 83%
# Save the correct rates so it doesn't have to be run every time
#save(correct_rates,
#     file=file.path('C:/Users/spenc/Documents/Kaggle Competitions/Titanic/rf_tuning_mat.Rdata'))


# XG Boost

# Prep data for XGBoost (I'm using Age1 here because it performed better than Age2 for the other models)
Xmat1 <- model.matrix(data=titanic,~Pclass+Sex+Age1+SibSp+Parch+Fare+Embarked+GivenCabin)
response <- data.matrix(titanic$Survived)
# Look at an initial model with predictions to test if working correctly
xgb_initial <- xgboost(data=Xmat1, label=response, 
                       nrounds=50,
                       eta=0.01,
                       max_depth=10,
                       objective = "binary:logistic",
                       verbose=0)
predict(xgb_initial, newdata=Xmat1[1:10,])

# leave ten out cross validation to choose optimal parameters for xgboost
# leave one out cross validation will take too much time
xgb_10crossVal <- function(dat,eta,max_depth,nrounds) {
  n <- floor(nrow(dat)/10)
  #pb <- txtProgressBar(min=0,max=n,style=3)
  the_vals <- numeric(nrow(dat))
  for (i in 1:n) {
    if (i==89) {
      rows_not_included <- ((i-1)*10+1):891
    } else {
      rows_not_included <- ((i-1)*10+1):(i*10)
    }
    dat_leave_out <- dat[-rows_not_included,]
    response_give <- response[-rows_not_included]
    xgb_fit <- xgboost(data=dat_leave_out, label=response_give, 
                       nrounds=nrounds,
                       eta=eta,
                       max_depth=max_depth,
                       objective = "binary:logistic",
                       verbose=0)
    the_vals[rows_not_included] <- predict(xgb_fit, newdata=data.matrix(dat[rows_not_included,]))
    #setTxtProgressBar(pb,i)
  }
  cutoff_vals <- seq(0,1,by=0.01)
  cutoff_means <- numeric(length(cutoff_vals))
  for (k in 1:length(cutoff_vals)) {
    cutoff_means[k] <- mean(as.numeric(the_vals > cutoff_vals[k]) == titanic$Survived)
  }
  best_cutoff <- cutoff_vals[which.max(cutoff_means)]
  best_percentage <- max(cutoff_means)
  list(best_percentage,best_cutoff)
}
testing_function <- xgb_10crossVal(Xmat1,0.001,10,100)


# Do a grid search to find the parameters that give the best prediction performance for xgboost
eta_checks <- c(0.001,0.1,0.3)
max_depth_checks <- c(2,5,10)
n_round_checks <- c(50,100,1000)
vals_to_check <- expand.grid(eta_checks,max_depth_checks,n_round_checks)

n_iterations <- nrow(vals_to_check)
prediction_to_save_xgb <- numeric(n_iterations)
for (j in 1:n_iterations) {
  prediction_to_save_xgb[j] <- xgb_10crossVal(Xmat1,eta=vals_to_check[j,1], max_depth=vals_to_check[j,2],nrounds=vals_to_check[j,3])[[1]]
  print(j)
}
# save files so I don't have to run again
#save(prediction_to_save_xgb,
#     file=file.path('C:/Users/spenc/Documents/Kaggle Competitions/Titanic/xgb_tuning_mat.Rdata'))

# Best tuning parameters and best prediction performance and best cutoff:0.5
vals_to_check[which.max(prediction_to_save_xgb),] # eta:0.001, max_depth:10, and nrounds:100
max(prediction_to_save_xgb) # 84.62%




# Now that I have fit all three models and found the correct prediction rate from cross validation,
#   I use XGBoost with tuning parameters eta:0.001, max_depth:10, and nrounds:100 to predict 
#   Survival on the test data because it had the best cross validated prediction rate.
final_xgb <- xgboost(data=Xmat1, label=response, 
                     nrounds=100,
                     eta=0.001,
                     max_depth=10,
                     objective = "binary:logistic",
                     verbose=0)

# fill in missing values of Age for testing data using linear regression model fit earlier
test.lm.missing <- predict(lm.missing,newdata=titanic.pred[is.na(titanic.pred$Age),])
titanic.pred$Age1 <- titanic.pred$Age
titanic.pred$Age1[is.na(titanic.pred$Age1)] <- test.lm.missing

test_matrix <- model.matrix(data=titanic.pred, ~Pclass+Sex+Age1+SibSp+Parch+Fare+Embarked+GivenCabin)
test_prediction_percentages <- predict(final_xgb, newdata=test_matrix)
test_predictions <- as.numeric(test_prediction_percentages > 0.5)

# Create cvs file for my predictions
test_csv <- data.frame(cbind(titanic.pred$PassengerId,test_predictions))
colnames(test_csv) <- c('PassengerId','Survived')
write.csv(test_csv,"titanic_final_predictions.csv",quote=FALSE,row.names=FALSE)


# Now I want to look at the variables that had the highest impact
# I mainly just want to look at the values where a woman was predicted to die and then where a man was predicted to survive
# I also look at a SHAP scores plot