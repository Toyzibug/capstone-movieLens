if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
library(lubridate)
library(stringr)
library(dplyr)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")
movielens
movielens%>%as.tibble()
# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Dataset description -- dataset in tibble format
as_tibble(edx)
##dimention of the training and testing data
dim(edx)
dim(validation)

# convert timestamp to date type
edx<-edx%>%mutate(date=as.Date(as.POSIXct(timestamp, origin="1970-01-01")))%>%select(-timestamp)

#extract the coming out year to a new column
year<-str_extract(edx$title,"\\(\\d{4}\\)")
year<-str_replace(year,"\\(","")
year<-str_replace(year,"\\)","")
edx<-edx%>%mutate(year=year)
##delete the year info from column "title"
title<-str_replace(edx$title,"\\(\\d{4}\\)", "")
edx<-edx%>%select(-title)%>%mutate(title=title)
as.tibble(edx)

#exploring the data --rating distribution
summary(edx$rating)
mu<-mean(edx$rating)

#exploring the data -- movie effect with number of rating
edx%>%group_by(movieId)%>%summarise(n=n())%>%
  ggplot(aes(n))+geom_histogram(bins = 30,color="black")+
  xlab("Number of Ratings")+
  ggtitle("Histogram - Number of Ratings per Movie")
rating_number<-edx%>%group_by(movieId)%>%summarise(n=n())
summary(rating_number$n)
## user effect with number of rating
edx%>%group_by(userId)%>%summarise(n=n())%>%
  ggplot(aes(n))+geom_histogram(bins=30,color="black")+
  xlab("Number of Ratings")+
  ggtitle("Histogram - Number of Ratings per User")

##training and testing dataset of edx before modelling
index<-createDataPartition(edx$rating,times=1,p=0.1,list=FALSE)
edxtrain<-edx[-index,]
tmp<-edx[index,]
###make sure the user and movie in train set apears also in test set
edxtest <- tmp %>% 
  semi_join(edxtrain, by = "movieId") %>%
  semi_join(edxtrain, by = "userId")

#modeling -- movie effect parameter bi
mu<-mean(edxtrain$rating)
bi<-edxtrain%>%group_by(movieId)%>%summarise(bi=mean(rating-mu))
## visualization of bi
bi%>%ggplot(aes(bi))+geom_boxplot()+ggtitle("Boxplot of bi")

#modelling -- user effect parameter bi
bu<-edxtrain%>%left_join(bi,by="movieId")%>%
  group_by(userId)%>%summarise(bu=mean(rating-bi-mu))
## visualization of bu
bu%>%ggplot(aes(bu))+geom_boxplot()+ggtitle("Boxplot of bu")

##modelling ---genres effect parameter bg
bg<-edxtrain%>%left_join(bi,by="movieId")%>%left_join(bu,by="userId")%>%group_by(genres)%>%
  summarise(bg=mean(rating-bi-bu-mu))


# Function to calculate the RMSE
RMSE <- function(true_ratings, pred){ 
  sqrt(mean((true_ratings - pred)^2))
}

#prediction of the first model
pred1<-edxtest%>%left_join(bi,by="movieId")%>%left_join(bg,by="genres")%>%
  left_join(bu,by="userId")%>%mutate(pred=mu+bi+bu+bg)%>%pull(pred)

## calculating the RMSE of the prediction
rmse1<-RMSE(pred1,edxtest$rating)
rmse1

##involving genres effect
bg<-edxtrain%>%left_join(bi,by="movieId")%>%left_join(bu,by="userId")%>%group_by(genres)%>%
  summarise(bg=mean(rating-bi-bu-mu))


#modeling -- parameters of the complex model and predicting with different lamda
lmda<-seq(0,10,0.25)

rmses<-sapply(lmda,function(l){
  mu<-mean(edxtrain$rating)
  
  bil<-edxtrain%>%
    group_by(movieId)%>%summarise(bil=sum(rating-mu)/(n()+l))
  
  bul<-edxtrain%>%left_join(bil,by="movieId")%>%
    group_by(userId)%>%summarise(bul=sum(rating-bil-mu)/(n()+l))
  
  bg<-edxtrain%>%left_join(bi,by="movieId")%>%left_join(bu,by="userId")%>%group_by(genres)%>%
    summarise(bgl=sum(rating-bi-bu-mu)/(n()+l))
  
  pred2<-edxtest%>%
    left_join(bil,by="movieId")%>%left_join(bul,by="userId")%>%left_join(bgl,by="genres")%>%
    mutate(pred=mu+bil+bul)%>%pull(pred)
  return(RMSE(pred2,edxtest$rating))
})

##visualize the lamda with its RMSE value
qplot(lmda,rmses)
lamda<-lmda[which.min(rmses)]
lamda

min(rmses)

#modeling -- parameter distribution compare with the best lamda 
bil<-edxtrain%>%
  group_by(movieId)%>%summarise(bil=sum(rating-mu)/(n()+4.5))
##boxplot the bi and bil for compare
bi_value<-bi$bi
bil_value<-bil$bil
b1<-data.frame(bi_value,bil_value)
par(mgp=c(3,2,0))
boxplot(b1,main="Compare of bi value")


##final test on validation dataset
mu<-mean(edx$rating)
bil<-edx%>%
  group_by(movieId)%>%summarise(bil=sum(rating-mu)/(n()+4.5))
bul<-edx%>%left_join(bil,by="movieId")%>%
  group_by(userId)%>%summarise(bul=sum(rating-bil-mu)/(n()+4.5))
bgl<-edx%>%
  left_join(bil,by="movieId")%>%left_join(bul,by="userId")%>%group_by(genres)%>%
  summarise(bgl=sum(rating-bil-bul-mu)/(n()+4.5))
pred3<-validation%>%
  left_join(bil,by="movieId")%>%left_join(bul,by="userId")%>%left_join(bgl,by="genres")%>%
  mutate(pred=mu+bil+bul+bgl)%>%pull(pred)
RMSE(pred3,validation$rating)


