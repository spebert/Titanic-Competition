
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



#### The analysis ####


# Since there are missing values for age I do logistic regression on whether age is missing or not to see if missingness is at random
titanic$MissingAge <- as.factor(as.numeric(is.na(titanic$Age)))

age.logit <- glm(MissingAge~Pclass+Sex+Survived+SibSp+Parch+Fare+Embarked+GivenCabin,data=titanic,family="binomial")
summary(age.logit)
table(titanic$MissingAge,titanic$Pclass)
# It seems as though missingness in age is lower for second class tickets and higher for number of siblings
# Since there is no significant relationship between survived and missingness for age I fit two models with
#   and without age included.

# Embarked has 2 missing values so I fill those values with S the most common one 
titanic$Embarked[which(titanic$Embarked=="")] <- "S"
titanic$Embarked <- as.factor(titanic$Embarked)

# Logistic regression
titanic$Pclass <- as.factor(titanic$Pclass)
titanic$GivenCabin <- as.factor(as.numeric(titanic$Cabin != ""))
table(titanic$Pclass,titanic$GivenCabin)
# Most of the people with a cabin number are first class so there is some colinearity there

logit.fit <- glm(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+GivenCabin,data=titanic,family="binomial")
summary(logit.fit)
predict(logit.fit,newdata = titanic[1,], type="response")

# logit leave one out cross validation function
logit_crossVal <- function(dat, cutoff=0.5) {
  n <- nrow(dat)
  the_vals <- numeric(n)
  for (i in 1:n) {
    dat_leave_out <- dat[-i,]
    if (is.na(dat$Age[i])) {
      # Fit model without age
      cv_logit_fit <- glm(Survived~Pclass+Sex+SibSp+Parch+Fare+Embarked+GivenCabin,data=dat_leave_out,family="binomial")
    } else {
      # fit model with age
      cv_logit_fit <- glm(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+GivenCabin,data=dat_leave_out,family="binomial")
    }
    the_vals[i] <- predict(cv_logit_fit, newdata=dat[i,], type="response")
  }
  list(correct_rate=mean((the_vals>cutoff)==dat$Survived),prediction_percentage=the_vals)
}

logistic_leave_one_out <- logit_crossVal(titanic)
# The correct prediction rate for leave one out cross validation is 81% for logistic regression
# Grid search for the best cutoff
cutoff_vals <- seq(0,1,by=0.01)
choose_cutoff <- sapply(cutoff_vals, function(x) mean( (logistic_leave_one_out$prediction_percentage>x)==titanic$Survived ))
cutoff_vals[which.max(choose_cutoff)]
# The best cutoff is at 0.55. This gives a correct prediction performance of 82%. 


# Random forests

titanic$Survived <- as.factor(titanic$Survived)
rf <- randomForest(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+GivenCabin,data=titanic,ntree=500,na.action = na.roughfix)
predict(rf,newdata=titanic[6,])
mean(predict(rf,newdata=titanic)==titanic$Survived)

# leave one out cross validation to choose optimal parameters in the random forest
rf_crossVal <- function(dat,ntree) {
  n <- nrow(dat)
  the_vals <- numeric(n)
  for (i in 1:n) {
    dat_leave_out <- dat[-i,]
    if (is.na(dat$Age[i])) {
      # Fit model without age
      rf_fit <- randomForest(Survived~Pclass+Sex+SibSp+Parch+Fare+Embarked+GivenCabin,data=dat_leave_out,ntree=ntree)
    } else {
      # fit model with age
      rf_fit <- randomForest(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+GivenCabin,data=dat_leave_out,ntree=ntree,na.action = na.omit)
    }
    the_vals[i] <- predict(rf_fit, newdata=dat[i,])
  }
  mean(the_vals-1==dat$Survived) # minus 1 here because the prediction is in 1 and 2
}
rf_crossVal(titanic,ntree=500)

# There are options to use imputation for the missing values. In this case I fit two models with and without age
#   na.omit fits the model with the data where age is given. Further analyses on the missing values would be
#   helpful but aren't covered here.

# Test different values for ntree to see which performs the best
ntree.test <- 2:10 * 100
correct_rates <- sapply(ntree.test, FUN=rf_crossVal, dat=titanic)

# Work in progress for xgboost section
# XG Boost
#xgbfit <- xgboost(data=data.matrix(titanic),)

#cbind(model.matrix(~-1+Pclass+Sex+SibSp+Parch+Fare+Embarked+GivenCabin,data=titanic),titanic$Age)
#xgb <- xgboost(data=data.matrix(suicide.train[,-106]),label=data.matrix(suicide.train[,106])-1,
#               nrounds=60,eta=0.7,objective = "binary:logistic")

