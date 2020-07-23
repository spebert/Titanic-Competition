
"""
The purpose of this file is to perform the same analysis I did in R for the 
titanic competition but instead using Python.
"""

# Import required packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read in train and test data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


#### EDA ####

train.head()
print(train.columns)
"""
Meaning for each column
  PassengerId     Unique ID of passenger
  Survived        (No=0, Yes=1)
  Pclass          Socio Economic status of passenger
  Name            Name of passenger
  Sex             male or female
  Age             Age of passenger (there are some NA values)
  SibSp           Number of siblings / spouses aboard
  Parch           Number of children / parents aboard
  Ticket          Ticket Number
  Fare            Passenger Fare
  Cabin           Cabin Number (Most values are empty)
  Embarked        Port of Embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)
"""

# Graphs for each variable

# Gender
male_survival = train[train["Sex"] == "male"]["Survived"].mean()
female_survival = train[train["Sex"] == "female"]["Survived"].mean()
plt.bar([0,1], [male_survival, female_survival], align="center", 
        alpha=0.5)
plt.xticks([0,1],["male","female"])
plt.xlabel("Gender")
plt.ylabel("Proportion Survived")
plt.title("Gender and Survival")
plt.show()

# Class
class_group = train.groupby("Pclass")["Survived"].mean()
plt.bar([0,1,2], class_group, align="center", 
        alpha=0.5)
plt.xticks([0,1,2],["1","2","3"])
plt.xlabel("Class")
plt.ylabel("Proportion Survived")
plt.title("Class and Survival")
plt.show()

# Siblings
sib_group = train.groupby("SibSp")["Survived"].mean()
plt.bar([0,1,2,3,4,5,6], sib_group, align="center", 
        alpha=0.5)
plt.xticks([0,1,2,3,4,5,6],["0","1","2","3","4","5","8"])
plt.xlabel("Siblings")
plt.ylabel("Proportion Survived")
plt.title("Siblings and Survival")
plt.show()

# Parent/Children
par_group = train.groupby("Parch")["Survived"].mean()
plt.bar([0,1,2,3,4,5,6], par_group, align="center", 
        alpha=0.5)
plt.xticks([0,1,2,3,4,5,6],["0","1","2","3","4","5","6"])
plt.xlabel("Parent/Child")
plt.ylabel("Proportion Survived")
plt.title("Parent/Child and Survival")
plt.show()

# Fare

