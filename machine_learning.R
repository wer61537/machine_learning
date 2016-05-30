library(caret)
library("xtable")
library(randomForest)
library(MASS)
library(xgboost)
library(chron)

# use parallel processing for faster execution (when hardware resources permit)
library(doParallel)

#remove all objects just to be safe
rm(list = ls(all = TRUE))
set.seed(1234)
setwd("c:/coursera/machine_learning/data/")

#helper function to calculate out of sample classification error
missClass = function(values, prediction) {
  sum(prediction != values)/length(values)
}


#detect number of available cores
cl <- makeCluster(detectCores()) 
#register the number of cores for parallel execution
#cl is too much
registerDoParallel(2)                    

# Download data files (once) and read them 
for (destfile in c("pml-training.csv", "pml-testing.csv")) 
  if(!file.exists(destfile)) 
    download.file(paste0("http://d396qusza40orc.cloudfront.net/predmachlearn/",destfile),
                  destfile=destfile, method="auto")
pml.train <- read.csv("pml-training.csv",na.strings=c("NA","#DIV/0!",""))
pml.test <- read.csv("pml-testing.csv",na.strings=c("NA","#DIV/0!",""))

str(pml.train)
str(pml.test)

head(pml.train)
head(pml.test)

names.remove1 <-c(
  "X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", 
  "new_window", "num_window") 

pml.train <- pml.train[,!names(pml.train) %in% names.remove1] 
pml.test <- pml.test[,!names(pml.test) %in% names.remove1] 

# helper function from Stephen Turner
#http://www.gettinggeneticsdone.com/2011/02/summarize-missing-data-for-all.html
propmiss <- function(dataframe) {
  m <- sapply(dataframe, function(x) {
    data.frame(
      nmiss=sum(is.na(x)), 
      n=length(x), 
      propmiss=sum(is.na(x))/length(x)
    )
  })
  d <- data.frame(t(m))
  d <- sapply(d, unlist)
  d <- as.data.frame(d)
  d$variable <- row.names(d)
  row.names(d) <- NULL
  d <- cbind(d[ncol(d)],d[-ncol(d)])
  return(d[order(d$propmiss), ])
}
#get the fraction missing by variable
names.remove2<-propmiss(pml.train)
c<-names.remove2[which(names.remove2$propmiss > 0.9), 1]
pml.train.clean<-pml.train[,!names(pml.train) %in% names.remove2[which(names.remove2$propmiss > 0.9), 1]] 
pml.test.clean<-pml.test[,!names(pml.test) %in% names.remove2[which(names.remove2$propmiss > 0.9), 1]] 
#recheck
propmiss(pml.train.clean)
propmiss(pml.test.clean)

dim(pml.train.clean)  #19622 and 53
dim(pml.test.clean) #20 and 53

#just to be sure, are all but the last column names the same?
#if(names(pml.train.clean[,-53]) == names(pml.test.clean[,-53]))
#  {
#    print ("True")
#  }
#

#do standard tests for 0 variance
zeroVarCols<-nearZeroVar(pml.train.clean, saveMetrics = TRUE)
pml.train.clean<-pml.train.clean[, zeroVarCols$nzv==FALSE]
pml.test.clean<-pml.test.clean[, zeroVarCols$nzv==FALSE]
#really no changes
dim(pml.train.clean)  #19622 and 53
dim(pml.test.clean) #20 and 53

# find columns to remove in order to reduce pair-wise correlations
highlyCorDescr <- findCorrelation(cor(pml.train.clean[,1:52]), cutoff = .75)
pml.train.clean <- pml.train.clean[,-highlyCorDescr]
pml.test.clean <- pml.test.clean[,-highlyCorDescr]
dim(pml.train.clean);dim(pml.test.clean);
#number of variables is now 32

#cast the classe column as a factor
pml.train.clean$classe = factor(pml.train.clean$classe)

#subset the pml.train.clean into a train and validation set, use 70/30
subset <- createDataPartition(y=pml.train.clean$classe, p=0.7, list=FALSE)
sub.training <- pml.train.clean[subset, ] 
sub.validation<- pml.train.clean[-subset, ]
dim(sub.training)
dim(sub.validation)
head(sub.training)
head(sub.validation)
#do the counts match 19622
dim(sub.training)[1] 
dim(sub.validation)[1]



#Regression Tree
#10 fold validaton
train_control <- trainControl(method="cv", number=10)
#rpart
rpart.model <- train(classe ~., method="rpart", data=sub.training, 
                     trControl = train_control)

rpart.confuse.train<- confusionMatrix(sub.training$classe, predict(rpart.model, newdata=sub.training))
rpart.confuse.validation<- confusionMatrix(sub.validation$classe, predict(rpart.model, newdata=sub.validation))
rpart.predict <-predict(rpart.model, newdata=sub.validation)
rpart.errRate = missClass(sub.validation$classe, rpart.predict)


#random forest
rf.model <- randomForest(classe ~., data=sub.training,method="class")

rf.confuse.train<- confusionMatrix(sub.training$classe, predict(rf.model, newdata=sub.training))
rf.confuse.validation<- confusionMatrix(sub.validation$classe, predict(rf.model, newdata=sub.validation))
rf.confuse.train
rf.confuse.validation
rf.predict <-predict(rf.model, newdata=sub.validation)
rf.errRate = missClass(sub.validation$classe, rf.predict)

#gbm takes a very long time
#gbm
gbm.model <- train(classe ~ ., data=sub.training, method="gbm")

gbm.confuse.train<- confusionMatrix(sub.training$classe, predict(gbm.model, newdata=sub.training))
gbm.confuse.validation<- confusionMatrix(sub.validation$classe, predict(gbm.model, newdata=sub.validation))
gbm.predict <-predict(gbm.model, newdata=sub.validation)
gbm.errRate = missClass(sub.validation$classe, gbm.predict)


#xgbTree
xgbTree.model <- train(classe ~ ., data=sub.training, method="xgbTree")

xgbTree.confuse.train<- confusionMatrix(sub.training$classe, predict(xgbTree.model, newdata=sub.training))
xgbTree.confuse.validation<- confusionMatrix(sub.validation$classe, predict(xgbTree.model, newdata=sub.validation))
xgbTree.predict <-predict(xgbTree.model, newdata=sub.validation)
xgbTree.errRate = missClass(sub.validation$classe, xgbTree.predict)


#lda
lda.model <- train(classe ~ ., data=sub.training, method="lda")

lda.confuse.train<- confusionMatrix(sub.training$classe, predict(lda.model, newdata=sub.training))
lda.confuse.validation<- confusionMatrix(sub.validation$classe, predict(lda.model, newdata=sub.validation))
lda.predict <-predict(lda.model, newdata=sub.validation)
lda.errRate = missClass(sub.validation$classe, lda.predict)

#knn  
#takes some time
knn.model <- train(classe ~ ., data = sub.training, method = "knn")
knn.confuse.train<- confusionMatrix(sub.training$classe, predict(knn.model, newdata=sub.training))
knn.confuse.validation<- confusionMatrix(sub.validation$classe, predict(knn.model, newdata=sub.validation))
knn.predict <-predict(knn.model, newdata=sub.validation)
knn.errRate = missClass(sub.validation$classe, knn.predict)

#clearly the rf is better

pml.test.predict <- predict(rf.model, pml.test, type="class")
pml.test.predict

pml_write_files = function(x){
  n = length(x)
  path <- "c:/coursera/machine_learning/data/answers"
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=file.path(path, filename),quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(pml.test.predict)


