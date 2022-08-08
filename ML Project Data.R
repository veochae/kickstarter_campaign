#script written by Veo Chae

#dataset import
project = read.csv("/Volumes/T7 Touch/Senior (2021)/Machine Learning/Final Project/kickstarter_data_full.csv")

#data structure
str(project)
summary(project)

#empty cell analysis
colSums(is.na(project))
  #empty cell is listed as "" therefore needs to be transformed to NA


#counting the number of "" within each column
out <- NULL
for (i in c(1:68)){
  matrix = sum(project[,i] == "")
  a <- as.data.frame(matrix)
  out <- rbind(out,a)
  
}
out

#row name <- project column name
naaaam <- colnames(project)
row.names(out) <- naaaam
out

#first setting all "" to NA
for (i in c(1:nrow(project))){
  for (t in c(1:68)){
    if (project[i,t] == "" || is.na(project[i,t]) == TRUE){
      project[i,t] <- NA
    }else{
      project[i,t] <- project[i,t]
    }
  }
}

#reassess empty cells count for each columns
colSums(is.na(project))

#deleting the columns that has the majority empty
project$friends = NULL
project$permissions = NULL
project$is_backing = NULL
project$is_starred = NULL

project <- project[complete.cases(project),]

str(project)

colSums(is.na(project))

#keeping only the "success" and "fail" cases
unique(project$state)

project_1 <- project[project$state == "successful" ,]
project_2 <- project[project$state == "failed" ,]
project_clean <- rbind(project_1, project_2)

project_clean


#exporting clean data to csv file
write.csv(project_clean, "/Volumes/T7 Touch/Senior (2021)/Machine Learning/Final Project/ML_Clean_Data.csv")

data <- read.csv("D:/Senior (2021)/Machine Learning/Final Project/Final Project/ML_Clean_Data_with_Sentiment.csv")
