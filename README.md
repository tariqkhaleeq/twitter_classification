# Twitter Sentiment Classification

I picked up sentiment140 dataset for this project. The dataset contains 1,600,000 tweets extracted using Twitter's API. The tweets have been annotated (0 = negative, 2 = neutral, 4 = positive). These annotation can be used to detect sentiments from these tweets. The data set contains the following six fields:

 **target**: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
 
 **ids**: The id of the tweet ( 2087)
 
 **date**: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
 
 **flag**: The query (lyx). If there is no query, then this value is NO_QUERY
 
 **user**: the user that tweeted (robotickilldozr)
 
 **text**: the text of the tweet (Lyx is cool)
    
  You can read more about the data from [here](http://help.sentiment140.com/for-students/). To read more about the details of the concept applied in gathering the dataset can be read [here](https://www.linkedin.com/pulse/social-machine-learning-h2o-twitter-python-marios-michailidis/)
  
  For this project I restricted myself to logistic regression. However, I would like to apply more ML techniques in the future if time permits. [Future inspiration](http://www.yuefly.com/Public/Files/2017-03-07/58beb0822faef.pdf)
