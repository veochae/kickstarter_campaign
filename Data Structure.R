#### This script has been written by Veo Chae 
#### for the purpose of data reformation for ML 2021 Team Kickstarters Final Project

install.packages("tidyverse")
library("tidyverse")

mydata=read.csv("/Volumes/T7 Touch/Senior (2021)/Machine Learning/Final Project/ML_Clean_Data_Final.csv")

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

#printing table of unique levels within the categorical variables
for (i in c(1:length(cat.vars))){
  print(cat.vars[i])
  print(table(mydata[,cat.vars[i]]))
}

#further, we nullified slug_sentiment as it is mostly neutral and does not add value

mydata$slug_sentiment <- NULL
str(mydata)

#using tidyverse fct function to lump the minimum counts into "other"
#from the analysis above, initially attempting to utilize function on "country", "category"
mydata$country <- fct_lump_min(mydata$country, 20)
mydata$category <- fct_lump_min(mydata$category, 40)
str(mydata)


#############################################################################
######################## ONLY FOR ENSEMBLE METHOD MODELS ################################
#standardizing numeric variables
num.vars <- names(Filter(is.numeric,mydata))


for(i in c(1:7)){
  for(j in c(1:nrow(mydata))){
    mydata[j,num.vars[i]] <- (mydata[j,num.vars[i]] - mean(mydata[,num.vars[i]]))/sd(mydata[,num.vars[i]])
  }
}

str(mydata)
#############################################################################
#############################################################################

mydata$id <- NULL



