# Some notes before starting:
#   * Read all the way through the instructions. 
# * Models must be built using Python, R, or SAS.
# * New features can be created.
# * Users cannot add or supplement with external data. 
# * While simple techniques may develop adequate models, success in this exercise typically involves feature engineering and model tuning.
# * Throughout your code, please use comments to document your thought process as you move through exploratory data analysis, feature engineering, model tuning, etc.  
# * Please review your submission against the submission expectations.
# 
# 
# Step 1 - Clean and prepare your data: 
#   There are several entries where values have been deleted to simulate dirty data. Please clean the data with whatever method(s) you believe is best/most suitable. Note that some of the missing values are truly blank (unknown answers).  Success in this exercise typically involves feature engineering and avoiding data leakage.
# 
# Step 2 - Build your models: 
#   Please use two different machine learning/statistical algorithms to develop a total of two models. Please include comments that document choices you make (such as those for feature engineering and for model tuning). 
# 
# Step 3 - Generate predictions:
#   Create predictions on the data in test.csv using each of your trained models.  The predictions should be the class probabilities for belonging to the positive class (labeled '1').  
# 
# Be sure to output a prediction for each of the rows in the test dataset (10K rows).  Save the results of each of your models in a separate CSV file.  Title the two files 'results1.csv' and 'results2.csv'.  A result file should each have a single column representing the output from one model (no header label or index column is needed). 
# 
# Step 4 - Compare your modeling approaches:
#   Please prepare a relatively short write-up comparing the pros and cons of the two algorithms you used (PDF preferred). As part of the write-up, please identify which algorithm you think will perform the best. For the best performing model, are there choices you made in the context of the exercise that might be different in a business context? How would explain to a business partner the concept that one model is better than the other?
#   
#   Step 5 - Submit your work: 
#   Your submission should consist of all the code used for exploratory data analysis, cleaning, prepping, and modeling (text, html, or pdf preferred), the two result files (.csv format - each containing 10,000 decimal probabilities), and your write-up comparing the pros and cons of the two modeling techniques used (text, html, or pdf preferred). Note: The results files should not include the original data, only the probabilities.
# 
# Your work will be scored on techniques used (appropriateness and complexity), evaluation of the two techniques compared in the write-up, model performance on the data hold out  - measured by AUC, and your overall code/comments.  The threshold for passing model performance is set high, expecting that model tuning and feature engineering will be used.  The best score of the two models submitted will be used.
# 
# Please do not submit the original data back to us. 

# install.packages("pillar")
# install.packages("dplyr")
# install.packages("tibble")
# install.packages("pdflatex")
# install.packages("ggpubr")
# install.packages("neuralnet")
# install.packages("ada")

library(pillar)
library(dplyr)
library(ggplot2)
# library(pdflatex)
library(reshape2)
library(ggcorrplot)
library(randomForest)
library(factoextra)
library(ggpubr)
library(neuralnet)
library(ada)

########################################################################################################################
# Importing the data
########################################################################################################################

SF_Train<-read.csv(file = "C:/Users/puj83/OneDrive/CV/Cases/InsuranceX/exercise_04_train.csv", header = T, sep = ",")
SF_Test<-read.csv(file = "C:/Users/puj83/OneDrive/CV/Cases/InsuranceX/exercise_04_test.csv", header = T, sep = ",")

names(SF_Train)
names(SF_Test)

# str(SF_Train)

# Below are factors in SF_Train

# $ x34: Factor w/ 11 levels "","bmw","chevrolet",..: 2 10 2 10 10 10 2 9 9 4 ...
# $ x35: Factor w/ 9 levels "","fri","friday",..: 5 9 6 8 9 8 9 6 8 9 ...
# $ x41: Factor w/ 37814 levels "","$0.03 ","$0.09 ",..: 21448 27346 24405 28719 1817 22615 22255 4083 1533 28524 ...
# $ x45: Factor w/ 10 levels "","-0.01%","-0.02%",..: 2 6 6 7 2 6 7 7 6 6 ...
# $ x68: Factor w/ 13 levels "","Apr","Aug",..: 13 7 7 2 3 3 7 7 3 10 ...
# $ x93: Factor w/ 4 levels "","america","asia",..: 3 3 3 3 3 3 2 3 3 3 ...

# str(SF_Test)

# Below are factors in SF_Test

# $ x34: Factor w/ 11 levels "","bmw","chevrolet",..: 2 10 2 10 10 10 2 9 9 4 ...
# $ x35: Factor w/ 9 levels "","fri","friday",..: 5 9 6 8 9 8 9 6 8 9 ...
# $ x41: Factor w/ 37814 levels "","$0.03 ","$0.09 ",..: 21448 27346 24405 28719 1817 22615 22255 4083 1533 28524 ...
# $ x45: Factor w/ 10 levels "","-0.01%","-0.02%",..: 2 6 6 7 2 6 7 7 6 6 ...
# $ x68: Factor w/ 13 levels "","Apr","Aug",..: 13 7 7 2 3 3 7 7 3 10 ...
# $ x93: Factor w/ 4 levels "","america","asia",..: 3 3 3 3 3 3 2 3 3 3 ...

# Only Numerics

SF_Train_Correlation_Var <- subset(SF_Train, select = -c(x34, x35, x41, x45, x68, x93, y))
SF_Train_Correlation_Var_Copy<-SF_Train_Correlation_Var
SF_Train_Correlation_Var1<- subset(SF_Train, select = -c(x34, x35, x41, x45, x68, x93))

SF_Validation <- subset(SF_Test, select = -c(x34, x35, x41, x45, x68, x93))
SF_Validation_Copy<-SF_Validation

set.seed(123)

########################################################################################################################
# Basic Feature Engineering
########################################################################################################################

#########################
# Training Set
#########################

# Checking missing values

sum(is.na(SF_Train_Correlation_Var))/prod(dim(SF_Train_Correlation_Var))
SF_Train_Correlation_Var %>% summarize_all(funs(sum(is.na(.)) / length(.)))

# Mean imputation

SF_Train_Correlation_Var[] <- lapply(SF_Train_Correlation_Var, function(x) {
  x[is.na(x)] <- mean(x, na.rm = TRUE)
  x
})

# Standardizing/normalizing data

SF_Train_Correlation_Var<-apply(SF_Train_Correlation_Var, MARGIN = 2, FUN = function(X) (X - min(X))/diff(range(X)))
SF_Train_Correlation_Var<-as.data.frame(SF_Train_Correlation_Var)

#########################
# Validation Set
#########################

# Checking missing values

sum(is.na(SF_Validation))/prod(dim(SF_Validation))
SF_Validation %>% summarize_all(funs(sum(is.na(.)) / length(.)))

# Mean imputation

SF_Validation[] <- lapply(SF_Validation, function(x) {
  x[is.na(x)] <- mean(x, na.rm = TRUE)
  x
})

# Standardizing/normalizing data

SF_Validation<-apply(SF_Validation, MARGIN = 2, FUN = function(X) (X - min(X))/diff(range(X)))
SF_Validation<-as.data.frame(SF_Validation)

########################################################################################################################
# PCA Analysis / Exploratory Data Analysis (Feature Extraction)
########################################################################################################################

# PCA on Training Data to determine potential predictors to test on validation set

# Arguments for princomp():
# x: a numeric matrix or data frame
# cor: a logical value. If TRUE, the data will be centered and scaled before the analysis
# scores: a logical value. If TRUE, the coordinates on each principal component are calculated

res.pca<-princomp(SF_Train_Correlation_Var, cor = FALSE, scores = TRUE)

#Visualize eigenvalues (scree plot). Show the percentage of variances explained by each principal component.

fviz_eig(res.pca)

# Graph of variables.
# Positive correlated variables point to the same side of the plot.
# Negative correlated variables point to opposite sides of the graph.

fviz_pca_var(res.pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)

# PCA Results

eig.val <- get_eigenvalue(res.pca)

# Results for Variables
res.var <- get_pca_var(res.pca)
# res.var$coord          # Coordinates
# res.var$contrib        # Contributions to the PCs
# res.var$cos2           # Quality of representation

quanti.sup <- SF_Train_Correlation_Var
# head(quanti.sup)

# Predict coordinates and compute cos2
quanti.coord <- cor(quanti.sup, res.pca$x)
quanti.cos2 <- quanti.coord^2
# Graph of variables including supplementary variables
p <- fviz_pca_var(res.pca)
fviz_add(p, quanti.coord, color ="blue", geom="arrow")

# Here we'll show how to calculate the PCA results for variables: coordinates, cos2 and contributions:

# var.coord = loadings * the component standard deviations
# var.cos2 = var.coord^2
# var.contrib. The contribution of a variable to a given principal component is (in percentage) : (var.cos2 * 100) / (total cos2 of the component)

var_coord_func <- function(loadings, comp.sdev){
  loadings*comp.sdev
}
# Compute Coordinates
#::::::::::::::::::::::::::::::::::::::::
loadings <- res.pca$loadings
sdev <- res.pca$sdev
var.coord <- t(apply(loadings, 1, var_coord_func, sdev))
head(var.coord[, 1:4])

# Compute Cos2
#::::::::::::::::::::::::::::::::::::::::
var.cos2 <- var.coord^2
head(var.cos2[, 1:4])

# Compute contributions
#::::::::::::::::::::::::::::::::::::::::
comp.cos2 <- apply(var.cos2, 2, sum)
contrib <- function(var.cos2, comp.cos2){var.cos2*100/comp.cos2}
var.contrib <- t(apply(var.cos2,1, contrib, comp.cos2))
Contributions<-var.contrib[, 1:4]
ggcorrplot(Contributions)

SF_Train_Correlation_Vars_Corr <- subset(SF_Train_Correlation_Var, select = c(x42, x90, x44, x69, x71, x0, x27, x58, x5, x37))
corr2<- round(cor(SF_Train_Correlation_Vars_Corr), 2)
ggcorrplot(corr2)

## set the seed to make your partition reproducible
set.seed(123)

# Histograms of variables

ggplot(SF_Train_Correlation_Var1, aes(x=x42)) +
  geom_histogram(binwidth=1, colour="red", fill="red")

ggplot(SF_Train_Correlation_Var1, aes(x=x90)) +
  geom_histogram(binwidth=1, colour="black", fill="black")

ggplot(SF_Train_Correlation_Var1, aes(x=x44)) +
  geom_histogram(binwidth=1, colour="green", fill="green")

ggplot(SF_Train_Correlation_Var1, aes(x=x69)) +
  geom_histogram(binwidth=1, colour="blue", fill="blue")

ggplot(SF_Train_Correlation_Var1, aes(x=x71)) +
  geom_histogram(binwidth=1, colour="brown", fill="brown")

ggplot(SF_Train_Correlation_Var1, aes(x=x0)) +
  geom_histogram(binwidth=1, colour="purple", fill="purple")

########################################################################################################################
# Applying Machine Learning Algorithms via Basic Feature Engineering (Feature Selection)
########################################################################################################################

# Algorithm #1: Random Forest

SF_Train_Correlation_Var1$y<-as.numeric(as.character(SF_Train_Correlation_Var1$y))
SF_Train_Correlation_Var1_y<-as.data.frame(SF_Train_Correlation_Var1$y)
SF_Train_Correlation_Var<-cbind(SF_Train_Correlation_Var, SF_Train_Correlation_Var1_y)

names(SF_Train_Correlation_Var)[names(SF_Train_Correlation_Var) == "SF_Train_Correlation_Var1$y"] <- "y"
SF_Train_Correlation_Var$y<-as.factor(SF_Train_Correlation_Var$y)

## 75% of the sample size
smp_size <- floor(0.80 * nrow(SF_Train_Correlation_Var))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(SF_Train_Correlation_Var)), size = smp_size)

train <- SF_Train_Correlation_Var[train_ind, ]
test <- SF_Train_Correlation_Var[-train_ind, ]

# for reproducibility

set.seed(123)

#train

rf1<-randomForest(y~ x90 + x44 + x42 + x69 + x71 +x0 + x27 + x58+ x5 + x37,
                  data = train, ntree = 500,
                  mtry = 4, importance = TRUE, na.action = na.omit)

print(rf1)

#test

rf2<-randomForest(y~ x90 + x44 + x42 + x69 + x71 +x0 + x27 + x58+ x5 + x37,
                  data = test, ntree = 500,
                  mtry = 4, importance = TRUE, na.action = na.omit)

print(rf2)

test$pred_randomForest<-predict(rf1, test)
test_rf_comparison<-test %>% select(y, pred_randomForest)

#Validation

SF_Validation$pred_randomForest<-predict(rf1, SF_Validation)

# Algorithm #2: Adaboost

train_var <- subset(train, select = c(x90, x44, x42, x69, x71, x0, x27, x58, x5, x37))
ind_Attr1<-names(train_var)

test_var <- subset(test, select = c(x90, x44, x42, x69, x71, x0, x27, x58, x5, x37))
ind_Attr2<-names(test_var)

# Build best ada boost model
ada1<-ada(x = train[,ind_Attr1],
            y = train$y,
            iter=20, loss="logistic",verbose=TRUE) # 20 Iterations

# Look at the model summary
summary(ada1)

# Build best ada boost model
ada2<-ada(x = test[,ind_Attr2],
          y = test$y,
          iter=20, loss="logistic",verbose=TRUE) # 20 Iterations

# Look at the model summary
summary(ada2)

# Predict on train data
pred_Train<-predict(ada1, train[,ind_Attr1])

# Build confusion matrix and find accuracy
cm_Train = table(train$y, pred_Train)
accu_Train= sum(diag(cm_Train))/sum(cm_Train)
rm(pred_Train, cm_Train)

# Predict on test data
pred_Test = predict(ada1, test[,ind_Attr2])

# Build confusion matrix and find accuracy
cm_Test = table(test$y, pred_Test)
accu_Test= sum(diag(cm_Test))/sum(cm_Test)
rm(pred_Test, cm_Test)

#Validation

SF_Validation$pred_ada<-predict(ada1, SF_Validation)

########################################################################################################################
# More Feature Engineering to Improve Model
########################################################################################################################

#########################
# Training Set
#########################

# Checking missing values

sum(is.na(SF_Train_Correlation_Var_Copy))/prod(dim(SF_Train_Correlation_Var_Copy))
SF_Train_Correlation_Var_Copy %>% summarize_all(funs(sum(is.na(.)) / length(.)))

# Mean imputation

SF_Train_Correlation_Var_Copy[] <- lapply(SF_Train_Correlation_Var_Copy, function(x) { 
  x[is.na(x)] <- mean(x, na.rm = TRUE)
  x
})

# Treating outliers by Winsorizing/Capping

# Winsorizing

fun <- function(x){
  quantiles <- quantile( x, c(.05, .95 ) )
  x[ x < quantiles[1] ] <- quantiles[1]
  x[ x > quantiles[2] ] <- quantiles[2]
  x
}

SF_Train_Correlation_Var_Copy_BP <- subset(SF_Train_Correlation_Var_Copy, select = c(x90, x44, x42, x69, x71, x0, x27, x58, x5, x37))
ggplot(data = melt(SF_Train_Correlation_Var_Copy_BP), aes(x=variable, y=value)) + geom_boxplot(aes(fill=variable))

SF_Train_Correlation_Var_Copy$x90_WC<-fun(SF_Train_Correlation_Var_Copy$x90)
SF_Train_Correlation_Var_Copy$x44_WC<-fun(SF_Train_Correlation_Var_Copy$x44)
SF_Train_Correlation_Var_Copy$x42_WC<-fun(SF_Train_Correlation_Var_Copy$x42)
SF_Train_Correlation_Var_Copy$x69_WC<-fun(SF_Train_Correlation_Var_Copy$x69)
SF_Train_Correlation_Var_Copy$x71_WC<-fun(SF_Train_Correlation_Var_Copy$x71)
SF_Train_Correlation_Var_Copy$x0_WC<-fun(SF_Train_Correlation_Var_Copy$x0)
SF_Train_Correlation_Var_Copy$x27_WC<-fun(SF_Train_Correlation_Var_Copy$x27)
SF_Train_Correlation_Var_Copy$x58_WC<-fun(SF_Train_Correlation_Var_Copy$x58)
SF_Train_Correlation_Var_Copy$x5_WC<-fun(SF_Train_Correlation_Var_Copy$x5)
SF_Train_Correlation_Var_Copy$x37_WC<-fun(SF_Train_Correlation_Var_Copy$x37)

SF_Train_Correlation_Var_Copy_BP2 <- subset(SF_Train_Correlation_Var_Copy, select = c(x90_WC, x44_WC, x42_WC, x69_WC, x71_WC, x0_WC, x27_WC, x58_WC, x5_WC, x37_WC))
ggplot(data = melt(SF_Train_Correlation_Var_Copy_BP2), aes(x=variable, y=value)) + geom_boxplot(aes(fill=variable))

# Capping

SF_Train_Correlation_Var_Copy$x42_WC[SF_Train_Correlation_Var_Copy$x42_WC > 200] = 200
SF_Train_Correlation_Var_Copy$x42_WC[SF_Train_Correlation_Var_Copy$x42_WC < -200] = 200

max(SF_Train_Correlation_Var_Copy$x42_WC)
min(SF_Train_Correlation_Var_Copy$x42_WC)

# Standardizing/normalizing data 

SF_Train_Correlation_Var_Copy<-apply(SF_Train_Correlation_Var_Copy, MARGIN = 2, FUN = function(X) (X - min(X))/diff(range(X)))
SF_Train_Correlation_Var_Copy<-as.data.frame(SF_Train_Correlation_Var_Copy)

# Bucketing/binning/categorization

SF_Train_Correlation_Var_Copy$x90_WC <- ifelse(SF_Train_Correlation_Var_Copy$x90_WC > 0.5, 1, 0)
SF_Train_Correlation_Var_Copy$x44_WC <- ifelse(SF_Train_Correlation_Var_Copy$x44_WC > 0.5, 1, 0)
SF_Train_Correlation_Var_Copy$x0_WC <- ifelse(SF_Train_Correlation_Var_Copy$x0_WC > 0.5, 1, 0)

# Interaction

SF_Train_Correlation_Var_Copy$Three_Var_mean <- rowMeans(subset(SF_Train_Correlation_Var_Copy, select = c(x90, x44, x0)), na.rm = TRUE)


#########################
# Validation Set
#########################

# Checking missing values

sum(is.na(SF_Validation_Copy))/prod(dim(SF_Validation_Copy))
SF_Validation_Copy %>% summarize_all(funs(sum(is.na(.)) / length(.)))

# Mean imputation

SF_Validation_Copy[] <- lapply(SF_Validation_Copy, function(x) { 
  x[is.na(x)] <- mean(x, na.rm = TRUE)
  x
})

# Treating outliers by Winsorizing/Capping

# Winsorizing

fun <- function(x){
  quantiles <- quantile( x, c(.05, .95 ) )
  x[ x < quantiles[1] ] <- quantiles[1]
  x[ x > quantiles[2] ] <- quantiles[2]
  x
}

SF_Validation_Correlation_Var_Copy_BP <- subset(SF_Validation_Copy, select = c(x90, x44, x42, x69, x71, x0, x27, x58, x5, x37))
ggplot(data = melt(SF_Validation_Correlation_Var_Copy_BP), aes(x=variable, y=value)) + geom_boxplot(aes(fill=variable))

SF_Validation_Copy$x90_WC<-fun(SF_Validation_Copy$x90)
SF_Validation_Copy$x44_WC<-fun(SF_Validation_Copy$x44)
SF_Validation_Copy$x42_WC<-fun(SF_Validation_Copy$x42)
SF_Validation_Copy$x69_WC<-fun(SF_Validation_Copy$x69)
SF_Validation_Copy$x71_WC<-fun(SF_Validation_Copy$x71)
SF_Validation_Copy$x0_WC<-fun(SF_Validation_Copy$x0)
SF_Validation_Copy$x27_WC<-fun(SF_Validation_Copy$x27)
SF_Validation_Copy$x58_WC<-fun(SF_Validation_Copy$x58)
SF_Validation_Copy$x5_WC<-fun(SF_Validation_Copy$x5)
SF_Validation_Copy$x37_WC<-fun(SF_Validation_Copy$x37)

SF_Validation_Correlation_Var_Copy_BP2 <- subset(SF_Validation_Copy, select = c(x90_WC, x44_WC, x42_WC, x69_WC, x71_WC, x0_WC, x27_WC, x58_WC, x5_WC, x37_WC))
ggplot(data = melt(SF_Validation_Correlation_Var_Copy_BP2), aes(x=variable, y=value)) + geom_boxplot(aes(fill=variable))

# Capping

SF_Validation_Copy$x42_WC[SF_Validation_Copy$x42_WC > 200] = 200
SF_Validation_Copy$x42_WC[SF_Validation_Copy$x42_WC < -200] = 200

max(SF_Validation_Copy$x42_WC)
min(SF_Validation_Copy$x42_WC)

# Standardizing/normalizing data 

SF_Validation_Copy<-apply(SF_Validation_Copy, MARGIN = 2, FUN = function(X) (X - min(X))/diff(range(X)))
SF_Validation_Copy<-as.data.frame(SF_Validation_Copy)

# Bucketing/binning/categorization

SF_Validation_Copy$x90_WC <- ifelse(SF_Validation_Copy$x90_WC > 0.5, 1, 0)
SF_Validation_Copy$x44_WC <- ifelse(SF_Validation_Copy$x44_WC > 0.5, 1, 0)
SF_Validation_Copy$x0_WC <- ifelse(SF_Validation_Copy$x0_WC > 0.5, 1, 0)

# Interaction

SF_Validation_Copy$Three_Var_mean <- rowMeans(subset(SF_Validation_Copy, select = c(x90, x44, x0)), na.rm = TRUE)

########################################################################################################################
# Machine Learning Algorithms After Advanced Feature Engineering (Feature Selection)
########################################################################################################################

# Algorithm #1: Random Forest 

SF_Train_Correlation_Var1$y<-as.numeric(as.character(SF_Train_Correlation_Var1$y))
SF_Train_Correlation_Var1_y<-as.data.frame(SF_Train_Correlation_Var1$y)
SF_Train_Correlation_Var_Copy<-cbind(SF_Train_Correlation_Var_Copy, SF_Train_Correlation_Var1_y)

names(SF_Train_Correlation_Var_Copy)[names(SF_Train_Correlation_Var_Copy) == "SF_Train_Correlation_Var1$y"] <- "y"
SF_Train_Correlation_Var_Copy$y<-as.factor(SF_Train_Correlation_Var_Copy$y)

## 75% of the sample size
smp_size <- floor(0.80 * nrow(SF_Train_Correlation_Var_Copy))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(SF_Train_Correlation_Var_Copy)), size = smp_size)

train <- SF_Train_Correlation_Var_Copy[train_ind, ]
test <- SF_Train_Correlation_Var_Copy[-train_ind, ]

# for reproducibility

set.seed(123)

#train

rf1<-randomForest(y~ x90 + x44 + x0 + x42 + x69 + x71 +      
                    x0 + x40 + x25 + x95 + x8 + x53 + x61 + x22 + x10 + x78 +      
                    x21 + x74 + x20 + x63 + x75 + x57 + x56 + x19 + x18 + x49 +      
                    x96 + x97 + x50 + x99 + x4 + x3 + x80 + x70 + x83 + x58 +      
                    x5 + x37 + x27 + x12 + x66,
                  data = train, ntree = 500,
                  mtry = 12, importance = TRUE, na.action = na.omit)

print(rf1)

#test

rf2<-randomForest(y~ x90 + x44 + x0 + x42 + x69 + x71 +      
                    x0 + x40 + x25 + x95 + x8 + x53 + x61 + x22 + x10 + x78 +      
                    x21 + x74 + x20 + x63 + x75 + x57 + x56 + x19 + x18 + x49 +      
                    x96 + x97 + x50 + x99 + x4 + x3 + x80 + x70 + x83 + x58 +      
                    x5 + x37 + x27 + x12 + x66,
                  data = train, ntree = 500,
                  mtry = 12, importance = TRUE, na.action = na.omit)
print(rf2)

test$pred_randomForest<-predict(rf1, test)
test_rf_comparison<-test %>% select(x90, x44, x0, x42, x69, x71,      
                                      x40, x25, x95, x8, x53, x61, x22, x10, x78,      
                                      x21, x74, x20, x63, x75, x57, x56, x19, x18, x49,      
                                      x96, x97, x50, x99, x4, x3, x80, x70, x83, x58,      
                                      x5, x37, x27, x12, x66, y, pred_randomForest)
test_rf_comparison$Misclassified <- ifelse(test_rf_comparison$y == test_rf_comparison$pred_randomForest, 1, 0)

#Validation

SF_Validation_Copy$pred_randomForest<-predict(rf1, SF_Validation_Copy)
write.csv(SF_Validation_Copy$pred_randomForest, "C:/Users/puj83/OneDrive/CV/Cases/InsuranceX/results1.csv")

# Algorithm #2: Adaboost

train_var <- subset(train, select = c(x90, x44, x0, x42, x69, x71,      
                                      x40, x25, x95, x8, x53, x61, x22, x10, x78,      
                                      x21, x74, x20, x63, x75, x57, x56, x19, x18, x49,      
                                      x96, x97, x50, x99, x4, x3, x80, x70, x83, x58,      
                                      x5, x37, x27, x12, x66))
ind_Attr1<-names(train_var)

test_var <- subset(test, select = c(x90, x44, x0, x42, x69, x71,      
                                    x40, x25, x95, x8, x53, x61, x22, x10, x78,      
                                    x21, x74, x20, x63, x75, x57, x56, x19, x18, x49,      
                                    x96, x97, x50, x99, x4, x3, x80, x70, x83, x58,      
                                    x5, x37, x27, x12, x66))
ind_Attr2<-names(test_var)

# Build best ada boost model 
ada1<-ada(x = train[,ind_Attr1], 
          y = train$y, 
          iter=100, loss="logistic",verbose=TRUE) # 100 Iterations 

# Look at the model summary
summary(ada1)

# Build best ada boost model 
ada2<-ada(x = test[,ind_Attr2], 
          y = test$y, 
          iter=100, loss="logistic",verbose=TRUE) # 100 Iterations 

# Look at the model summary
summary(ada2)

# Predict on train data  
pred_Train<-predict(ada1, train[,ind_Attr1])  

# Build confusion matrix and find accuracy   
cm_Train = table(train$y, pred_Train)
accu_Train= sum(diag(cm_Train))/sum(cm_Train)
rm(pred_Train, cm_Train)

# Predict on test data
pred_Test = predict(ada1, test[,ind_Attr2]) 

# Build confusion matrix and find accuracy   
cm_Test = table(test$y, pred_Test)
accu_Test= sum(diag(cm_Test))/sum(cm_Test)
rm(pred_Test, cm_Test)

#Validation

SF_Validation_Copy$pred_ada<-predict(ada1, SF_Validation_Copy)
write.csv(SF_Validation_Copy$pred_ada, "C:/Users/puj83/OneDrive/CV/Cases/InsuranceX/results2.csv")



