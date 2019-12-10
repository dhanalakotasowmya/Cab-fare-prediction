#remove all program/objects from RAM
rm(list=ls())

#set wirking directory
setwd("D:/Car Fare Prediction/")

#checking current working directory
getwd()

#before starting this project with dataset. Initially I open test & train_cab files in excel and just look the data available
# starting the project I need to load some libraries to deal with this data
install.packages(c("lattice", "dplyr", "plyr", "ggplot2", "corrgram" ))

#load train & test dataset for analysis
train = read.csv("train_cab.csv", header = T)
test = read.csv("test.csv", header =T)

#understanding the data by getting number of observations and variables
dim(train)
dim(test)
#in R studio I see that, train data has 16067 observations and 7 variables
# whereas test data has 9914 observations and 6 variables

#getting structure of both train and test dataset
str(train)
str(test)

#cheking few data in train dataset
head(train,10)

#cheking few data in test dataset
head(test, 10)

#After checking both the dataset, 
#I found that there is 6 indepedent variable and 1 dependent variable in train dataset. 
#So I need to analyse the train dataset properly and develope the model 
#and that model will be implemented on test dataset.

#print concise summary of train dataset
summary(train)

#print concise summary of test dataset
summary(test)

#Seeing the both dataset summary, 
#I conclude that, there are some missing values in train dataset 
#i.e. fare_amount and passenger_count and No missing data in test dataset.
#for that reason I need to do the Missing Value Annalysis here

#before missing values I need to convert factor data into numeric to perform missing value operations
#but in fare_amount I found that there are some missing values but these are not showing as NA
#for that I convert fare_amount into vector then numeric
train$fare_amount = as.vector(train$fare_amount)

#convert fare_amount from vector to numeric
train$fare_amount = as.numeric(train$fare_amount)

#convert pickup_datetime
train$pickup_datetime = as.POSIXct(train$pickup_datetime, tz = "UTC",format ="%Y-%m-%d %H:%M:%OS")

#############################################################################################
#Missing Value Analysis
##############################################################################################
#checking numeber of missing values in train dataset
sum(is.na(train)) #found 81 missing values in train dataset

#checking variables with missing values count
sapply(train, function(x) sum(is.na(x)))

#there are 25 missing values in fare_amount, 1 missing value in pickup_date
#and 55 missing values in passenger_count

#checking number of missing values in test dataset
sum(is.na(test)) #no missing value in test dataset

#create data frame with missing percentage
missing_val = data.frame(apply(train,2, function(x){sum(is.na(x))}))

#convert row names into columns
missing_val$columns = row.names(missing_val)
row.names(missing_val) = NULL

#rename the variable names
names(missing_val)[1] = "Missing_Percentage"

#calculate percentage
missing_val$Missing_Percentage = (missing_val$Missing_Percentage / (nrow(train)))*100

#after missing percentage calculations 
#there is only 0.15% and 0.34% missing values present in fare_amount and passenger_count resp.
#dropping the missing values from dataset
train = na.omit(train)

#############################################################################################
# Outlier Analysis
#############################################################################################
#checking if any negative & zero fare_amount present in dataset
sum(train$fare_amount<=0)

#4 fare_amount has negative & zero value, which wont make any sense. Remove this fields
train = subset(train, train$fare_amount>0) #4 observations dropped

#summary of fare_amount
summary(train$fare_amount)

#this shows that there is fare_amount Min = 0.01 but Max = 54343 which is not possible for Car Fare
#to understand the highest values better I arrange this train dataset in decending order
train = train[order(-train$fare_amount),]

#After Checking highest fare_amount, fount that there 3 values which are highest and not possible 
#in fare_amount i.e. fare-amount = 54343.00, fare_amount = 4343.0000 and fare_amount = 0.01. 
#I need to drop these values from the train dataset
train = subset(train,train$fare_amount<4000) #2 observations dropped

train = subset(train, train$fare_amount!=0.01) #1 observation drop

#now there is highest fare_amount is 453.00
#and lowest fare_amount is 1.14

#Now deal with passenger_count
summary(train$passenger_count)

#Checking the passenger_count summary. 
#I shock to see the max value is 5345. Lol which is not possible even in train. 
#Lets check the list of passenger_count in descending order. It will clear the picture
train = train[order(-train$passenger_count),]

#After Checking, passenger_count ranges from 35 to 5345. 
#As I am analyzing dataset of Cabs, then This is not possible. 
#Also there are some observations with passenger_count = 0 & 0.12
#So these values are stictly outliers. I need to drop them.
#for cab fare prediction, I am consider the Max Number of passengers are 6. 
#Which make sense if cab is "SUV"

#Now drop the observations which has passenger_count value greater than or equal to 6
train = subset(train, train$passenger_count<=6) #19 observations dropped
train = subset(train, train$passenger_count!=0) #57 observations dropped
train = subset(train, train$passenger_count!=0.12) #1 observarion dropped

train = train[order(train$passenger_count),]

#lets check these thing with test dataset
summary(test)

#our test dataset is free from passenger_count outliers

#now its time to deal with the pickup longitude and latitude
##let us explore the pickup latitude and longitudes
summary(train$pickup_latitude)

summary(train$pickup_longitude)

#Exploring the pickup longitute and latidude it is difficult to know the location 
#for that reason I just Google these things. 
#Found one useful website to identify the location
#https://www.latlong.net/Show-Latitude-Longitude.html
#The above website gives the Actual Location based on longitude and latitude
#after doing so randon check on website and using describe 
#function, I conclude that the pilot project run in New York City of United States.

#Googleing gives the New York City Longitude and Latitude i.e.
#New York City Latitude and longitude coordinates are: 40.730610, -73.935242.

#after using describe function on train dataset pickup_longitude and pickup_latitude 
#are near to the actual cordinates of New York city. 
#For that I am considering the longitude and latitude ranges for our train dataset
#Latitude ranges from 39 to 43
#Longitude ranges from -72 to -76

#from that I can say there are clearly shows some outliers. Let's drop them

train = subset(train, train$pickup_latitude>39) #318 observations dropped
train = subset(train, train$pickup_latitude<43) #1 observations dropped

#similar operation for pickup longitude
train = subset(train, train$pickup_longitude < -72) #No observations dropped
train = subset(train, train$pickup_longitude>-76) #No observations dropped

#Now dealing with the dropoff_lattitude and dropoff_longitude
#similar operation for dropoff latitude and longitude
summary(train$dropoff_latitude)
summary(train$dropoff_longitude)

#dropping outliers in dropoff_longitude
train = subset(train, train$dropoff_latitude>39) #10 observations dropped
train = subset(train, train$dropoff_latitude<43) #No observation dropped

#similar operation for dropoff_longitude
train = subset(train, train$dropoff_longitude < -72) #03 observations dropped
train = subset(train, train$dropoff_longitude>-76) #No observations dropped

#############################################################################################
#feature selection
#############################################################################################
#Correlation Analysis: Here I am generating correlation matrix to understand 
#how the each variable related with each other. 
#In that I am plotting correlation matrix and 
#generate plot for better understanding 

library(corrgram)
#Correlation plot
corrgram(train, order = F, upper.panel = panel.pie, text.panel = panel.txt, main = "Correlation Plot")

#The above correlation analysis shows that, 
#each variable in dataset is indepedent and not correlated with each other. 
#So, each variable or feature play an important role to predict the fare_amount

###############################################################################################
#Data types conversions
###############################################################################################
#Check the data types of each columns in train and test dataset before further analysis
#check data types of train dataset 
sapply(train, class)

#check data types of test dataset
sapply(test, class)

#pickup_datetime seem to be datetime column which are in factor format. 

#similar way convert datetime in test dataset
test$pickup_datetime = as.POSIXct(test$pickup_datetime, tz = "UTC",format ="%Y-%m-%d %H:%M:%OS")

#in test dataset passenger_count variables in interger convert it  in numeric format
test$passenger_count = as.numeric(test$passenger_count)

################################################################################################
# Exploratory Data Analysis
#list assumptions I Made

#Does the passenger_count (Number of Passengers) affect the fare_amount (fare)?
#Does the pickup_datetime (Pickup Date & Time) affect the fare_amount (fare)?
#Does the day of the week affect the fare_amount (fare)?
#Does the distance travelled affect the fare_amount (fare)?
################################################################################################
#First, Let's split the datetime field pickup_datetime to the following:

#year
#month
#date
#hour
#day of the week

#after this conversion, I will calculate the day of the week and 
#come to our conclusion about how the pickup_location affect the fare_amount

train$Year = as.numeric(format(train$pickup_datetime, format = "%Y"))
train$Month = as.numeric(format(train$pickup_datetime, format = "%m"))
train$Date = as.numeric(format(train$pickup_datetime, format = "%d"))
#Monday = 1 Tuesday =2 , ....... Sunday =7
train$Day_of_week = as.numeric(strftime(train$pickup_datetime, format = "%u"))
train$Hour = as.numeric(format(train$pickup_datetime, format = "%H"))

#simiarly I am implementing this code for test data
test$Year = as.numeric(format(test$pickup_datetime, format = "%Y"))
test$Month = as.numeric(format(test$pickup_datetime, format = "%m"))
test$Date = as.numeric(format(test$pickup_datetime, format = "%d"))
#Monday = 1 Tuesday =2 , ....... Sunday =7
test$Day_of_week = as.numeric(strftime(test$pickup_datetime, format = "%u"))
test$Hour = as.numeric(format(test$pickup_datetime, format = "%H"))

#again check the datatypes of train & test dataset
sapply(train, class)
sapply(test, class)
#################################################################################################
#Distance calculations
#################################################################################################
#Now the most important, need to calculate the 
#distance travelled by cab from the pickup latutude & longitude and dropoff latutude & longitude. 
#To know how to calulate the distance from give data, I need to find formula.

#After Googleing, I found formula, named as Haversine Formula.

#Also creating a new filed 'distance' to fetch the distance between pickup and drop location

#We can calulate the distance in a sphere when latitudes and longitudes are given 
#by Haversine formula
#haversine(??) = sin²(??/2)

#Eventually, the formual boils down to the following where ?? is latitude, ?? is longitude, R is earth's radius (mean radius = 6,371km) to include latitude and longitude coordinates (A and B in this case).

#a = sin²((??B - ??A)/2) + cos ??A . cos ??B . sin²((??B - ??A)/2)

#c = 2 * atan2( ???a, ???(1???a) )

#d = R ??? c

#d = Haversine distance

#R have packages geosphere has built in function Haversine formula to calculate Distance 

install.packages("geosphere")
library(geosphere)

#Haversine Formula in train dataset
train$Distance = as.numeric(distHaversine(cbind(train$pickup_longitude, train$pickup_latitude), cbind(train$dropoff_longitude, train$dropoff_latitud), r = 6371))


#distance formula on test dataset
test$Distance = as.numeric(distHaversine(cbind(test$pickup_longitude, test$pickup_latitude), cbind(test$dropoff_longitude, test$dropoff_latitud), r = 6371))

#Here I also varify the actual distance I get through this formula and 
#distance provided by website: https://www.geodatasource.com/distance-calculator 
#based on longitude and latitude. 

#From this I conclude that I am on right track

#Now lets clarify assumptions one by one that I made earlier:

###########################################################################################
#Does the passenger_count (Number of Passengers) affect the fare_amount (fare)?
###########################################################################################
library(ggplot2)

#plot passenger_count vs frequency, 
ggplot(train, aes_string(x= train$passenger_count))+
  geom_histogram(fill = "cornsilk", colour = "black", binwidth = 1)+theme_bw()+
  xlab("Number of Passengers") + ylab("Frequency") 

#plot passenger_count vs fare_amount
plot(x = train$passenger_count,y = train$fare_amount, 
     xlab = "Number of Passengers", 
     ylab = "Fare",
     main = "passenger_count vs fare")

### from above two graphs, 
#I see that single passenger is most frequent travellers, 
#and highest fare also seems to come from cabs which carry just the 1 passenger.

##############################################################################################
#Does the pickup_datetime (Pickup Date & Time) affect the fare_amount (fare)?          
##############################################################################################
#does the pickupdate of month affect the fare_amount
plot(x = train$Date,y = train$fare_amount, 
     xlab = "Date", 
     ylab = "Fare",
     main = "Date vs Fare")

#From the above graph, the fare_amount throughout the month is seem to be uniform, 
#with maximum fare received on the 3rd

#does pickuptime of the day affect the fare_amount?
ggplot(train, aes_string(x= train$Hour))+
  geom_histogram(fill = "cornsilk", colour = "black", binwidth = 1)+theme_bw()+
  xlab("Hour of the day") + ylab("Frequency") 

#As the above graph shows, The time of the day plays an important role. 
#The frequency of cab rides seem to be lowest at 5AM and the highest at 6PM.
plot(x = train$Hour,y = train$fare_amount, 
     xlab = "Hour of the day", 
     ylab = "Fare",
     main = "Hour vs Fare")

#The fares, seem to be the high between 5AM to 10AM and 1PM to 4PM. 
#Maybe people who leave early to avoid traffic and cover large distance

################################################################################################
#Does the day of the week affect the fare_amount (fare)?                    
################################################################################################
ggplot(train, aes_string(x= train$Day_of_week))+
  geom_histogram(fill = "cornsilk", colour = "black", binwidth = 1)+theme_bw()+
  xlab("Day of Week") + ylab("Frequency") 

#The day of the week doesn't seem to have that much effect on number of cab rides
plot(x = train$Day_of_week,y = train$fare_amount, 
     xlab = "Day of Week", 
     ylab = "Fare",
     main = "Weekday vs Fare")
#above graph visualization not much affected by the day of week, The highest fares 
#seem to be on Monday and Friday and the lowest on Thrusday and Sunday. 
#Maybe people travel far distances on Monday to reach offices and Friday to reach back home, 
#hence there is high fares. 
#Many people preffer to stay at home on Sunday or has low fare due to holiday of offices shut.

#################################################################################################
#Does the distance travelled affect the fare_amount (fare)?
#################################################################################################
#This is the obevious answer and I am confident about it that the distance travelled 
#absolutly affect the fare_amount. But I will check with visualization.
ggplot(train, aes_string(x= train$Distance))+
  geom_histogram(fill = "cornsilk", colour = "black", binwidth = 1)+theme_bw()+
  xlab("Distance in Kms") + ylab("Frequency") 

plot(x = train$Distance,y = train$fare_amount, 
     xlab = "Distance in Kms", 
     ylab = "Fare",
     main = "Distance in Kms vs Fare")
#There are values which are greater than 100Kms! As this data is from New York City, 
#I am not sure why people would take a cab to travel more than 100Kms. 
#Since there is not much data present beyond 50Kms. So this is the outliers.

#In next step I am droping these outliers
train = subset(train, train$Distance<50) #11 observations dropped

##############################################################################################
#Further Outliers in Distance
###############################################################################################

#Now here, I found that some distances are showing 0 value. 
#Now I am checking the rows where the distance values are 0

sum(train$Distance==0)

#After evaluating I found that there are 155 observations with distance = 0.

#This could be the reason because:

##The cab waited the whole time and passenger cancelled the trip after sometime. 
##Thats the reason why the pickup and dropoff coordinates are same and maybe, 
##passenger was charged for the waiting time.

##otherwise, there may be the chances of wrong coordinates are entered.

##comparing the count of 155 observations with train dataset 15560. 
#I can drop this data, by condering the wrong coordnates are entered.

train = subset(train, train$Distance!=0) #155 observations dropped

#let's check if any rows with distance values with 0 in test dataset
sum(test$Distance==0)

#droping the observations distance =0 in test dataset
test = subset(test, test$Distance!=0) #85 observations dropped

###############################################################################################
#Modelling & Predictions
###############################################################################################

#Finally, Finally, Data Cleaning is done! Now lets bulid the model and predict the results

#In machine Learning there is Two main types:

#Supervised Machine Learning : knowledge of output. Target Variable is fix
#Unsupervised Machine Learning: No knowledge of Output. Self Guided Learnig Algorithms.

#Selecting model is main Part of Modelling, We have various model algorithms some of 
#the basic algorithms are:

#Linear Regression : Best suitable for Regression Model
#Logistic Regression: Suitable for Classification Model
#Decision Tree: Best suitable for Regression & Classification model
#Random Forest: Mostly used for Classification model analysis but can be use for Regression model
#KNN algorithms: Can be used for Regression and Classification model
#Naive Bayes: used for Classification Model

#Currently I am dealing with Regression Model So I am considering following algorithm models:

#Linear Regression
#Decision Tree
#Random Forest
#KNN Algorithms
#checking columns in train & test dataset
colnames(train)
colnames(test)

##drop the pickup_datetime columns as datetime cannot be directly used for modeling
#so this can done on both train and test dataset
install.packages("dplyr")
library(dplyr)

train = select(train, -c(pickup_datetime))
test = select(test, -c(pickup_datetime)) #actual test data which I use later 

#Now to predict the model, to simulate a train and test set, I am going to split randomly 
#this train dataset into 80% train and 20% test    

#Random Sample indexes
train_index = sample(1:nrow(train), 0.8*nrow(train))
test_index = setdiff(1:nrow(train), train_index)

#Build train_X, train_y, test_X, test_y
train_X = train[train_index, -1]
train_y = train[train_index, "fare_amount"]

test_X = train[test_index, -1]
test_y = train[test_index, "fare_amount"]


#################################################################################################
#Linear Regression Model
#################################################################################################
install.packages("usdm")
library(usdm)
model_LR = lm( train_y ~ ., data=train_X)

summary(model_LR)

#predict the output for test_X dataset
predict_LR = predict(model_LR, test_X) 

#few things i learn from this model
#the much important is p-value of the model. Here we get p-value less than 0.05 that why, 
#I am rejecting the null hypothesis and consider the alternate hypotheis
#means the variables in train dataset are associated to predict the fare_amount value
#but there are some variable which are not useful to predict the fare_value that we determine
#with the help of t-value (higher is better)
#as we see, drooff_latitude, passenger_count, date has lower t-value values that why 
#they are not associatied with fare_amount. 
#Other Variables are important mainly Distance has higher t-value

#R-Squared Value = 0.5754 which is good but We don't necessarily discard a model 
#based on a R-Squared value.

#####Better Solution is###########
#Do model Evaluation based on the Error Metrics for Regression:

#For classification problems, we have only used classification accuracy as our evaluation metric.
#But here we used Error Metrics to evaluate the model

#Mean Absolute Error (MAE): is the mean of the absolute value of the errors: 
#In [0,???), the smaller the better

#Mean Squared Error (MSE): is the mean of the squared errors: In [0,???), the smaller the better

#Mean Absolute Percent Error (MAPE): is the mean of the absolute percent value of the errors: 
#In [0,1), the smaller the better

#Root Mean Squared Error (RMSE) :is the square root of the mean of the squared errors: 
#In [0,???), the smaller the better

#Let's calculate these by hand, to get an intuitive sense for the results:
install.packages("Metrics")

library(Metrics)

# calculate MAE, MSE, MAPE, RMSE
mae(test_y, predict_LR) #2.153135
mse(test_y, predict_LR) #13.82451
mape(test_y, predict_LR) #0.2162309
rmse(test_y, predict_LR) #3.718132

####
#MAE gives less weight to outliers means it is not sensitive to outliers.
#MAPE is similar to MAE, but normalized the true obeservations. When true observation is zero 
#then this metric will be problematic
#MSE is a combination measurement of bias and variance of predictions. It is more popular.
#RSME is square Root of MSE, Root square is taken to make the units of the error be the same as 
#the units of the target. This measure gives more weight to large deviations such as outliers, 
#since large differences squared become larger and small (smaller than 1) 
#differences squared become smaller.

#Selection: Outoff these 4 error metrics, MSE and RMSE are mainly used for Time-Series dataset. 
#As I know, current working data is not a time dependend or time-series data.

#for that Reason the Model Evaluation is based on MAPE Error Metrics

################################################################################################
#Decision Tree
################################################################################################
#load libraries
library(rpart)
library(MASS)

#use rpart for Decision Tree regression 
model_DT = rpart(train_y ~ ., data=train_X, method = "anova")

summary(model_DT)

#predict the output for test_X dataset
predict_DT = predict(model_DT, test_X) 

# calculate MAE, MSE, MAPE, RMSE
mae(test_y, predict_DT) #2.581969
mse(test_y, predict_DT) #18.36605
mape(test_y, predict_DT) #0.2577115
rmse(test_y, predict_DT) #4.285564

################################################################################################
#Random Forest
################################################################################################
#load libraries
library(randomForest)

#use randomForest for Random Forest regression 
model_RF = randomForest(train_y ~ ., data=train_X, importance = TRUE, ntree = 500)

summary(model_RF)

#predict the output for test_X dataset
predict_RF = predict(model_RF, test_X) 

# calculate MAE, MSE, MAPE, RMSE
mae(test_y, predict_RF) #1.951885
mse(test_y, predict_RF) #15.31187
mape(test_y, predict_RF) #0.2085321
rmse(test_y, predict_RF) #3.913038

################################################################################################
#KNN Algorithms
################################################################################################
#load library
library(caret)

#use knnreg for KNN regression algorithms
model_KNN = knnreg(train_y ~.,data=train_X, k=5)

summary(model_KNN)

#predict the output for test_X dataset
predict_KNN = predict(model_KNN, test_X) 

# calculate MAE, MSE, MAPE, RMSE
mae(test_y, predict_KNN) #2.520559
mse(test_y, predict_KNN) #18.97416
mape(test_y, predict_KNN) #0.2618314
rmse(test_y, predict_KNN) #4.355934

#################################################################################################
#Selecting Best suitable model for final analysis
#################################################################################################
#After evaluating the MAPE, RSME values of all the model,
#I am consering the MAPE for model evaluatiomn becasue, it 
#calculate average absolute percent error for each time period minus actual values 
#divided by actual values. 
#Reason why I am going with MAPE over RSME, because during spliting of train & test data, this method select
#random data, if you execute splitting code multiple time, the RSME shows more variation compared 
#to MAPE

#Random Forest Model has smallest error metrics i.e.
#MAPE = 0.2085321

#So, for further analysis I am selecting Random Forest Model.

#################################################################################################
#Implementing Selected Model on actual Test Dataset
#################################################################################################
#checking structure of test dataset
dim(test)

#write the cleaned test file in system
write.csv(test, "test_final.csv", row.names = F)

#implementing RamdomForest model on test dataset
pred_test = predict(model_RF, test)

#################################################################################################
#Submission
#################################################################################################
#final Result Submission
df = read.csv("test_final.csv", header = T)
df$fare_amount = pred_test

#write the final output in system
write.csv(df, "submission_RF.csv", row.names = F)
