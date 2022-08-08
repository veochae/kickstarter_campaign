options(scipen = 999)
library(gmodels)
library(e1071)
library(pROC)
library(tree)
library(ftsa)

############### HOLDOUT METHOD################
numpredictors=dim(mydata)[2]-1

numfac=0

for (i in 1:numpredictors) {
  if ((is.factor(mydata[,i]))){
    numfac=numfac+1} 
}

nobs=dim(mydata)[1]
RNGkind(sample.kind = "Rejection")
set.seed(1) #sets the seed for random sampling

prop = prop.table(table(mydata$myresponse))
length.vector = round(0.8*nobs*prop)
train_size=sum(length.vector)
test_size=nobs-train_size
class.names = as.data.frame(prop)[,1]
numb.class = length(class.names)
resample=1

while (resample==1) {
  
  train_index = c()
  
  for(i in 1:numb.class){
    index_temp = which(mydata$myresponse==class.names[i])
    train_index_temp = sample(index_temp, length.vector[i], replace = F)
    train_index = c(train_index, train_index_temp)
  }
  
  mydata_train=mydata[train_index,] #randomly select the data for training set using the row numbers generated above
  mydata_test=mydata[-train_index,]#everything not in the training set should go into testing set
  
  right_fac=0 #denotes the number of factors with "right" distributions (i.e. - the unique levels match across mydata, test, and train data sets)
  
  for (i in 1:numpredictors) {
    if (is.factor(mydata_train[,i])) {
      if (setequal(intersect(as.vector(unique(mydata_train[,i])), as.vector(unique(mydata_test[,i]))),as.vector(unique(mydata[,i])))==TRUE)
        right_fac=right_fac+1
    }
  }
  
  if (right_fac==numfac) (resample=0) else (resample=1)
  
}  

test_predictors=mydata_test
test_predictors$myresponse=NULL
myresponse_test=as.data.frame(mydata_test$myresponse)
colnames(myresponse_test)="myresponse"
str(test_predictors)
str(myresponse_test)

dim(mydata_test) #confirms that testing data has only 20% of observations
dim(mydata_train) #confirms that training data has 80% of observations


########################################################
#NAIVE BAYES
########################################################

model=naiveBayes(myresponse ~ .,data=mydata_train)
pred=predict(model, test_predictors)
tbl=as.data.frame(table(myresponse_test$myresponse,pred))

for_export=cbind(mydata_test,pred)
for_export$Naive_Bayes_Classification=for_export$pred
for_export$pred=NULL

########################################################
#LOGISTIC REGRESSION
########################################################
cutoff = 0.5
logistic_fit=glm(formula=myresponse~.,family=binomial,data=mydata_train)  #This fits the logistic regression model

predicted=predict(logistic_fit, mydata_test, type="response") #Predicts the probabilities in the testing set

predicted1=as.data.frame(predicted)
id=as.numeric(rownames(predicted1))
predicted_tbl0=cbind(id,predicted1)
if ("id" %in% names(mydata_test)==FALSE) {mydata_test=cbind(id, mydata_test)}
predicted_tbl=merge(predicted_tbl0,mydata_test, by.x="id", all.x=T)
predicted_tbl$predicted_class=as.numeric(predicted_tbl$predicted>=cutoff)

########################################################
#CART
########################################################

tree_type="C"

#Enter the minimum number of items in each leaf

min_leaf_size=30

#Enter the minimum deviance for a node to be considered for a further split

min_deviance=0

full_tree=tree(myresponse ~ .,split="deviance",mindev=min_deviance, mincut=min_leaf_size,data=mydata_train)
#End growing the reference tree


#START 10-FOLD CROSS-VALIDATION TO FIND THE SIZE THAT THE FULL TREE WILL BE REDUCED TO

b_list=rep(1,100)

for (i in 1:100){
  
  set.seed(i)
  cv_tree=cv.tree(full_tree,K=10)
  cv_tree$size
  cv_tree$dev
  bestsize=min(cv_tree$size[cv_tree$dev==min(cv_tree$dev)])
  b_list[i]=bestsize
  #plot(cv_tree, type="p")
  
}

mytable=as.data.frame(table(b_list))
mytable_s=mytable[order(mytable$Freq),]
final_tree_size=as.numeric(paste(mytable_s[dim(mytable_s)[1],1]))
#END K-FOLD CROSS-VALIDATION TO FIND THE SIZE THAT THE FULL TREE WILL BE REDUCED TO


#START REDUCING THE FULL TREE TO OPTIMAL SIZE AND PLOTTING

bestcut=prune.tree(full_tree,best=final_tree_size)
plot(bestcut, type=c("uniform"))
text(bestcut, cex=0.6, digits = max(nchar(mydata$myresponse))+3)

if (tree_type=="R"){
  print(bestcut, digits=max(nchar(mydata$myresponse))+3)} else
    print(bestcut)



#END REDUCING THE FULL TREE TO OPTIMAL SIZE AND PLOTTING


#START PREDICTING THE RESPONSE IN THE TESTING SET (20 % SUBSET)

predicted=predict(bestcut,newdata=mydata_test, type="vector")
if (tree_type=="R") {
  id=as.numeric(names(predicted))
  temp_tbl=cbind(id,as.data.frame(predicted))
  mydata_test2=cbind(as.numeric(rownames(mydata_test)),mydata_test)
  colnames(mydata_test2)[1]="id"
  pred_table=merge(temp_tbl, mydata_test2, by.x="id", all.x=T)
  predicted=pred_table$predicted
  pred_table$predicted=NULL
  final_table=cbind(pred_table,predicted)
  final_table$id=NULL } else {
    
    predicted=as.data.frame(predicted)
    
    new.col = c()
    
    for(i in 1:(dim(predicted)[1])){
      
      find.max=which(predicted[i,]==max(predicted[i,]))
      
      #If there is a tie, assign a class randomly
      if (length(find.max)>1) {
        
        find.max=sample(find.max,1, replace=F)
        
      }
      
      new.col = c(new.col, names(predicted)[find.max])
    }
    
    
    predicted$predicted=new.col
    
    id=as.numeric(rownames(predicted))
    newdat=as.data.frame(id)
    newdat$predicted=predicted$predicted
    mydata_test2=cbind(as.numeric(rownames(mydata_test)),mydata_test)
    colnames(mydata_test2)[1]="id"
    final_table=merge(mydata_test2,newdat,by.x="id", all.x=T)}

print(bestcut)
printcp(bestcut)

importance.tbl=as.data.frame(unlist(rf.train.final$importance))
if (tree_type=="C") (which.col="MeanDecreaseAccuracy") else (which.col="%IncMSE")
q.09=quantile(importance.tbl[,which.col], 0.9)
most.important.predictors=rownames(importance.tbl)[which(importance.tbl[,which.col]>=q.09)]





##################################################3
naive_result <- as.data.frame(for_export$Naive_Bayes_Classification)
naive_result$`for_export$Naive_Bayes_Classification`<- as.character(naive_result$`for_export$Naive_Bayes_Classification`)
naive_result$`for_export$Naive_Bayes_Classification` <- replace(naive_result$`for_export$Naive_Bayes_Classification`,as.character(naive_result$`for_export$Naive_Bayes_Classification`) == "failed",0)
naive_result$`for_export$Naive_Bayes_Classification` <-replace(naive_result$`for_export$Naive_Bayes_Classification`,as.character(naive_result$`for_export$Naive_Bayes_Classification`) == "successful",1)

logistic_result <- as.data.frame(predicted_tbl$predicted_class)
logistic_result$`predicted_tbl$predicted_class` <- as.character(logistic_result$`predicted_tbl$predicted_class`)
logistic_result$`predicted_tbl$predicted_class` <- replace(logistic_result$`predicted_tbl$predicted_class`,as.character(logistic_result$`predicted_tbl$predicted_class`) == "failed",0)
logistic_result$`predicted_tbl$predicted_class` <-replace(logistic_result$`predicted_tbl$predicted_class`,as.character(logistic_result$`predicted_tbl$predicted_class`) == "successful",1)


cart_result <- as.data.frame(newdat$predicted)
cart_result$`newdat$predicted` <- as.character(cart_result$`newdat$predicted`)
cart_result$`newdat$predicted` <- replace(cart_result$`newdat$predicted`,as.character(cart_result$`newdat$predicted`) == "failed",0)
cart_result$`newdat$predicted` <-replace(cart_result$`newdat$predicted`,as.character(cart_result$`newdat$predicted`) == "successful",1)

myresponse_test$myresponse <- as.character(myresponse_test$myresponse)
myresponse_test$myresponse <- replace(myresponse_test$myresponse,as.character(myresponse_test$myresponse) == "failed",0)
myresponse_test$myresponse <-replace(myresponse_test$myresponse,as.character(myresponse_test$myresponse) == "successful",1)


FINAL <- cbind(naive_result,logistic_result,cart_result,myresponse_test)
FINAL$`for_export$Naive_Bayes_Classification` <- as.numeric(FINAL$`for_export$Naive_Bayes_Classification`)
FINAL$`predicted_tbl$predicted_class` <-as.numeric(FINAL$`predicted_tbl$predicted_class`)
FINAL$`newdat$predicted` <- as.numeric(FINAL$`newdat$predicted`)
FINAL$myresponse <- as.numeric(FINAL$myresponse)
for (i in c(1:nrow(FINAL))){
  if (FINAL[i,1]+FINAL[i,2]+FINAL[i,3] > 1){
    FINAL[i,5] <- 1
  } else {
    FINAL[i,5] <- 0
  }
}

for (i in c(1:nrow(FINAL))){
  if(FINAL[i,4] == FINAL[i,5]){
    FINAL[i,6] <- TRUE
  } else {
    FINAL[i,6] <- FALSE
  }
}

write.csv(FINAL, "/Volumes/T7 Touch/Senior (2021)/Machine Learning/Final_2.csv")
