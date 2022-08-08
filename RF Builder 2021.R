
####################################################################################################################

####################################################################################################################

####################################################################################################################

#############################This program was created by Davit Khachatryan##########################################
######©2016-2017 by Davit Khachatryan.  All rights reserved. No part of this document may  be reproduced or#########
######transmitted in any form or by any means, electronic, mechanical, photocopying, recording or otherwise#########
############################without prior written permission of Davit Khachatryan###################################

####################################################################################################################

####################################################################################################################

####################################################################################################################


#########################################################################################
#########################################################################################
###################################ATTENTION#############################################
#######################THIS MACRO IS INTENDED FOR RANDOM FORESTS#########################
#########################################################################################
#########################################################################################

options(scipen=999)
library(randomForest)
library(pdp) #to get the partial dependence plots on probability scale for
             #classification problems
library(gmodels)
library(ggplot2)

#############################################################################################
##############################UPDATE THE SECTION BELOW#######################################
#############################################################################################

#START OF SETUP

  #Enter "R" for a regression tree and "C" for a classification tree below.

  tree_type="C"
  
  #Enter the maximum allowable number of trees in the forest
  
  num.tree=500


#END OF SETUP


#START OF DATA IMPORT

#update the path below to point to the directory and name of your data in *.csv format  

  mydata=read.csv("C:/Users/bgilarde1/Desktop/rawdata/ML_Clean_Data_Final.csv")
  

  install.packages("tidyverse")
  library("tidyverse")
  
  str(mydata)
  
  #changing dependent variable to myresponse
  mydata$myresponse <- mydata$state
  mydata$state <- NULL
  mydata$myresponse <- as.factor(mydata$myresponse)
  
  #we do not need index, therefore null
  #we do not need usd_pledged, therefore null
  #name, blurb, slug information --> sentiment and length all extracted. 
  #Therefore, since we have id as identification of each kickstarter, we null the three variables as they do not add value to our anaysis
  #Nullifying variables that do not add value when stakeholders = user prior to launching his/her campaign
  
  mydata[,c("X","usd_pledged", "name","blurb","slug","backers_count","TOPCOUNTRY", "deadline", "launched_at", "deadline_yr", "launched_at_yr", "launched_to_state_change_days", "LaunchedTuesday", "staff_pick", "DeadlineWeekend")] <- NULL
  
  str(mydata)
  
  #changing variables into categorical variables
  mydata[,c("country", "category","deadline_weekday", "launched_at_weekday", "deadline_month", "deadline_day", 
            "deadline_hr", "sentiment", "blurb_sentiment", "slug_sentiment","launched_at_month","launched_at_day","launched_at_hr")] <- lapply(mydata[,c("country", "category","deadline_weekday", "launched_at_weekday", "deadline_month", "deadline_day", 
                                                                                                                                                         "deadline_hr", "sentiment", "blurb_sentiment", "slug_sentiment","launched_at_month","launched_at_day","launched_at_hr")],factor)
  str(mydata)
  
  ### For categorical variables, lumping the levels that has less than ____ counts in order to ensure 80/20 split is done appropriately
  
  #gathering names of the variables that are categorical
  cat.vars <- names(Filter(is.factor,mydata))
  
  mydata$slug_sentiment <- NULL
  mydata$id=NULL
#END OF REDUNDANT VARIABLE REMOVAL

#############################################################################################
#####################################ATTENTION###############################################
#############################################################################################

#######################IF THE ABOVE MODIFICATIONS ARE MADE CORRECTLY,########################
####AT THIS POINT "MYDATA" DATA FRAME SHOULD CONTAIN ONLY THE PREDICTORS AND THE OUTCOME.#### 
####IN CASE IT CONTAINS ANYTHING MORE OR LESS, THE CODE BELOW WILL NOT FUNCTION PROPERLY.####
#############################################################################################

str(mydata) #make sure the structure of your data reflects all the modifications made above

#############################################################################################
############################################################################################
#################HIGHLIGHT AND RUN THE CODE BELOW AND RUN####################################
#####################UNTIL "END OF FOREST SIZE FINDER"########################################

#START DATA BREAKDOWN FOR HOLDOUT METHOD

#Start finding the categorical predictors

numpredictors=dim(mydata)[2]-1

numfac=0

for (i in 1:numpredictors) {
  if ((is.factor(mydata[,i]))){
    numfac=numfac+1} 
}

#End finding the number of categorical predictors 

nobs=dim(mydata)[1]



if (tree_type=="R") {
  
  #Below is the setup for stratified 80-20 holdout sampling for a Regression Tree
  
  train_size=floor(0.8*nobs)
  test_size=nobs-train_size

} else {
  
  #Below is the setup for stratified 80-20 holdout sampling for a Classification Tree
  
  prop = prop.table(table(mydata$myresponse))
  length.vector = round(nobs*0.8*prop)
  train_size=sum(length.vector)
  test_size=nobs-train_size
  class.names = as.data.frame(prop)[,1]
  numb.class = length(class.names)}
  
  
resample=1
RNGkind(sample.kind = "Rejection")
set.seed(1) #sets the seed for random sampling

while (resample==1) {
    
    
    if (tree_type=="C") {
      
    train_index = c()
    
    for(i in 1:numb.class){
      index_temp = which(mydata$myresponse==class.names[i])
      train_index_temp = sample(index_temp, length.vector[i], replace = F)
      train_index = c(train_index, train_index_temp)
    }} else {
      train_index=sample(nobs,train_size, replace=F)
    }
    
    mydata_train=mydata[train_index,] #randomly select the data for training set using the row numbers generated above
    mydata_test=mydata[-train_index,]#everything not in the training set should go into testing set
    
    right_fac=0 #denotes the number of factors with "right" distributions (i.e. - the unique levels match across mydata, test, and train data sets)
    
    
    for (i in 1:numpredictors) {
      if (is.factor(mydata_train[,i])) {
        if (sum(as.vector(unique(mydata_test[,i])) %in% as.vector(unique(mydata_train[,i])))==length(unique(mydata_test[,i])))
          right_fac=right_fac+1
      }
    }
    
    if (right_fac==numfac) (resample=0) else (resample=1)
    
  }
  
dim(mydata_test) #confirms that testing data has only 20% of observations
dim(mydata_train) #confirms that training data has 80% of observations

#################################################################################
#################################################################################

#END DATA BREAKDOWN FOR HOLDOUT METHOD

#START FOREST SIZE FINDER
set.seed(123)#don't modify the seed
rf.train=randomForest(myresponse~., 
                      data=mydata_train,
                      ntree=num.tree,
                      importance=TRUE, na.action = na.omit)

ylim.ceiling=max(plot(rf.train))+0.20*(max(plot(rf.train))-min(plot(rf.train)))
ylim.floor=min(plot(rf.train))
plot(rf.train, main="Error Rate vs Number of Trees In the Forest",ylim=c(ylim.floor,ylim.ceiling))

if (tree_type=="C"){
rndF1.legend <- colnames(rf.train$err.rate)
legend("top",cex =0.7, legend=rndF1.legend, lty=rep.int(2,length(rndF1.legend)), col=c(1:length(rndF1.legend)), horiz=T)}

#################################################################################
#################################################################################
#END FOREST SIZE FINDER



#START FINAL CONFIGURATION

#Visually inspect the graph titled "Error Rate vs Number of Trees In the Forest"
#and identify a point on the horizontal axis where the error rate tends to stabilize
#Input that number for "num.tree.final" below

#################################################################################
num.tree.final=100
#################################################################################

#END FINAL CONFIGURATION

#################################################################################
#####################DO NOT MODIFY BEYOND THIS POINT#############################
#################################################################################
set.seed(123)#don't modify the seed
rf.train.final=randomForest(myresponse~., 
                      data=mydata_train,
                      ntree=num.tree.final,
                      importance=TRUE, na.action = na.omit)

varImpPlot(rf.train.final, type=1, scale=FALSE)#this will produce mean decrease in accuracy variable importance plot
                                         #for details https://bigdata.unl.edu/documents/ASA_Workshop_Materials/Why%20and%20how%20to%20use%20random%20forest%20variable%20importance%20measures.pdf
                                         #            https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1796903/
                                         #            https://www.displayr.com/how-is-variable-importance-calculated-for-a-random-forest/

#Finding the most important predictors for which partial dependence plots will be plotted
importance.tbl=as.data.frame(unlist(rf.train.final$importance))
if (tree_type=="C") (which.col="MeanDecreaseAccuracy") else (which.col="%IncMSE")
q.09=quantile(importance.tbl[,which.col], 0.9)
most.important.predictors=rownames(importance.tbl)[which(importance.tbl[,which.col]>=q.09)]

#partial dependence plots
if (tree_type=="C"){
  class.to.plot=rf.train.final$classes[1]
  title=paste("PD Plot for Class", class.to.plot)
  y.legend="Average Probability"}else 
    {title="PD Plot" 
    y.legend="Average Value of Outcome"}

pd.plot <-function (x) {partial(rf.train.final, x, plot = TRUE, prob=TRUE, quantiles=F,
              plot.engine = "ggplot2")+ggtitle(title)+ylab(y.legend)}

lapply(most.important.predictors, pd.plot)

#START PREDICTING THE RESPONSE IN THE TESTING SET (20 % SUBSET)
predictions=predict(rf.train.final, newdata = mydata_test)
mydata_test_w_predictions=cbind(mydata_test, predictions)

#Measuring predictive accuracy below

if (tree_type=="R") {
  
  
  abs.diff=abs(mydata_test_w_predictions$predictions-mydata_test_w_predictions$myresponse)
  mape=100*mean(abs.diff/abs(mydata_test_w_predictions$myresponse))
  rmse=sqrt(mean(abs.diff^2))
  
  print(paste("MAPE for Testing Set Is:", 
              round(mape,2)))
  
  print(paste("RMSE for Testing Set Is:", 
              round(rmse,2)))
  
  
} else {
                print("Confusion Matrix Is:")
                CrossTable(mydata_test_w_predictions$myresponse,mydata_test_w_predictions$predictions,prop.chisq=F,prop.t=F) }

#END PREDICTING THE RESPONSE IN THE TESTING SET (20 % SUBSET)

#############################################################################################
##############################THIS IS THE END OF THE MACRO###################################
#############################################################################################


