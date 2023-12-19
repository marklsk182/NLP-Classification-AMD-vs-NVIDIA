
# Project 3: Web APIs &NLP

### Introduction

The AMD strategic PLanning (consumer GPUs) team wants our team to find out whether the 'negative stigma' associated with AMD steams from genuine issues of it is a simply matter of perception


### Background 

Nvidia's product were well known for graphic processing solutions and it were common used for AI and deep learning systems. 

AMD produces a wide range of microchip products. AMD also well known for graphic processors, system processors, RAM memory and hard drives. However, AMD was often regarded as a budget brand although thier product were still closly match up with Nvidia high end products. 

Depsite constant effort that AMD has put in to improve, market still perceive AMD was a weaker brand as compared to Nvidia. On top of that, AMD user has been constantly complain about the driver issues of AMD.

### Problem Statement

It is unclear whether the perception of AMD product being problematic is a marketing or an engineering issue? If this negative stigma continues to drag for AMD, it will definitely affect the bottom line of business to the point that the company unable to make a profit.   


### Approaches that we going to take

In order to gain more insight, we are going to scrap the posts and comments from Reddit for both AMD and Nvidia. For both AMD and Nvidia subreddits, negative posts such as post with cursing lauguage without specific reference to any of the problem, the post will be removed. Therefore, we would only expect there to be negative posts about the actual problems. 

With this mechanism in place and using sentiment analysis, if it is an engineering issue, we would expect to see many more negative posts on r/amd than r/nvidia. If the negative posts are similar, then it is likely a marketing issue. 

### Objectives

- We want to build a model that has the ability to differentiate between comments about AMD and nvidia for widerspread deployment e.g PC forum (in the event that becomes necessary) 

- Using sentiment analysis on the same data set, the result could be used to determine whether is it a marketing issue or an engineering issue? so AMD can deploy resources to improve the perception of the product

### Data

- Titles, body text, comments posted to r/amd and r/nvidia

### Methods and Tool: 

1. TfidfVectorizer 
2. CountVectorizer 
3. Naive Bayes 
4. Random Forest 
5. Logistic Regression 
6. RidgeClassifier

### Success Metrics

1. F1 score for classification, we want to minimise both false positive and false negatives. This will be used to select the classification model for deployment on GPU discussion forums. 

2. Sentiment Analysis scores, will use this score to recommend, to either deploy more resources to marketing or engineering. 

# 1.0 Scraper  Mechanism

### 1.1.0 Initialize PRAW

- to get the client_id and client_secret can use the link below 
- https://www.reddit.com/prefs/apps/

### 1.2.0 Load from the previous file (where I last left off from scraping) and retrieve the last_post_id 

Using the function below, to retrieve out the last post id where I left off from. 

### 1.3.0 Scraping Mechanism

Using the function below, for each request it will only scrap 600 posts, after scraping 600 posts it will save in both formate Json or Pickle. 

while scraping from reddit, it will only scrap the informations that I wanted in the extract_post_data functions as shown below. It will also scrap every single comment from each post as shown in extract_comment_data. 

We can set the number of post that we wanted to scrap at "post_count", after scraping each post the program will rest for 1sec then it will continue the next scrap. 

Once it has scrap 600 posts, the program will save all the posts to Josn and Pickle and sleep for 10sec, before it starting to scrap again.

# 2.0 Preprocessing Text 

### 2.1.0 Converting from Json to pandas dataframe for Nvidia

#### 2.1.1 Preprocessing text for Nvidia dataframe

Preprocssing of text as the steps shown below 
- Removing URLS, emoji, dollar signs, newline characters. 
- Converting text to lower characters. 
- Using NLTK to perform stopwords removal and lemmatization

Perform word count and lenght of content count after preprocessing.  
 

### 2.2.0 Converting from Json to pandas dataframe for Amd

#### 2.2.1 Preprocessing text for Amd dataframe

Preprocssing of text as the steps shown below 
- Removing URLS, emoji, dollar signs, newline characters. 
- Converting text to lower characters. 
- Using NLTK to perform stopwords removal and lemmatization
 

### 2.3.0 Combining Amd and Nvidia into one dataframe 


# 3.0 Exploratory data analysis 

### 3.1.0 Nvidia and Amd Class  

Seem like it the class is imbalance, going to deal with it the modeling stage.  

3.2.0 Sentiment analysis compound Score for both Nvidia and Amd
VADER

#### 3.2.1 Sentiment analysis observation

- Sentiment mean value score: For Nvidia, the 0.207, while Amd was 0.137. This shows that Amd post generally were more negative as compare to Nvidia

### 3.3.0 Readability score for Amd and Nvidia 

#### 3.3.1 Observation of readability score
So both Amd and Nvidia had roughly the same readbility score, the contents of both posts were more suitable for adult audience. 

### 3.4.0 Correlation analysis for Nvidia

#### 3.4.1 Finding for nvidia using heatmap

**Content lenght and sentiment:** There is a slight positive correlation between content length and sentiment. This suggest that the longer post might be slightly more positive, but the correlation is weak. 


#### 3.4.2 Correlation analysis for Amd

#### 3.4.3 Finding for Amd using heatmap

**Content lenght and sentiment:** There is a slight positive correlation between content length and sentiment. This suggest that the longer post might be slightly more positive, but the correlation is weak. 


#### 3.4.4 Conclusion for both Amd and Nvidia using heatmap

For both Amd and Nvidia posts, they seem to share similar finding when we use heatmap to perform an analysis on them. This could conclude that both of Amd and Nvidia posts are very similar.  

### 3.5.0 Using CountVectorizer (cvec) to show the top 10 unigram bigram trigram

#### 3.5.1 Nvidia  top 10 unigram bigram trigram

#### 3.5.2Nvidia Top 10 words

#### 3.5.3 Amd  top 10 unigram bigram trigram

#### 3.5.4 Amd Top 10 words

#### 3.5.5 Summary of Nvidia and Amd Top 10 common words

### 3.6.0 Using TfidfVectorizer (tvec) to show the top 10 unigram bigram trigram

#### 3.6.1 Nvidia  top 10 unigram bigram trigram

#### 3.6.2 Nvidia Top 10 words

#### 3.6.3 Amd  top 10 unigram bigram trigram

#### 3.6.4 Amd Top 10 words

#### 3.6.5 Summary of Nvidia and Amd Top 10 common words

### 3.7.0 Analyzing clean_content with less than 2 word count 

#### 3.7.1 Analyzing the clean_content with 1 word count  

#### 3.7.2 Analyzing the  clean_content with 2 word count

#### 3.7.3 After analyzing clean_content with 1 or 2 word count. 

# 4.0 Modeling 

### 4.1.0 Finding the Best Mode to used CountVectorizer (cvec)

#### 4.1.1 Conclusion from baseline model using cvec 

From the results above, it shows that some models were performs better as compare to the others, So I am going to choose the following model to perform grid search

**MultinomialNB** has the closest range in of train and test score as compare to the rest of the model. So this model has been choosen for grid search.   

**LogisticRegression** The range difference between the train and test score was about 0.11. Althought it is much wider as compare to MultinomialNB, but will still choose this model to perform grid search, due to it perform much more better as compare to the rest of the model. 

**RidgeClassifier**  Will use this model, since this model able to give me the list of top 20 best correlated features.  

**Random Forest** Although it perform badly wihtout any tuning, but will use this model for gridsearch, since this model are able to handle overfitting. 

### 4.2.0 CountVectorizer Top 20 Occurings Words 

### 4.3.0 Selected model using cvec for GridSearch

#### 4.3.1.0 MultinomialNB using cvec

#### 4.3.1.1 MultinomialNB cvec Classification Report

#### 4.3.2.0 LogisticRegression using cvec

#### 4.3.2.1  LogisticRegression cvec Classification Report

#### 4.3.2.2  LogisticRegression cvec Top 20 Features 

#### 4.3.3.0 RidgeClassifier using cvec

#### 4.3.3.1 RidgeClassifier cvec Classification Report

#### 4.3.3.2 RidgeClassifier cvec Top 20 Features

#### 4.3.4.0 Random Forest using cvec 

#### 4.3.4.1 Random Forest cvec Classification Report

#### 4.3.4.2 Random Forest cvec Top 20 Features

### 4.4.0 Modeling using TfidfVectorizer (tvec)

#### 4.4.1 Conclusion from baseline model using  tvec 

### 4.5.0 CountVectorizer Top 20 Occurings Words  

#### 4.6.1.0 MultinomialNB using tvec

#### 4.4.1.1 MultinomialNB tvec Classification Report

#### 4.6.2.0  LogisticRegression using tvec

#### 4.6.2.1 LogisticRegression tvec Classification Report

#### 4.6.2.2  LogisticRegression tvec Top 20 features 

#### 4.6.2.3  LogisticRegression cvec Top 20 Features

#### 4.6.3.0  RidgeClassifier using tvec

#### 4.6.3.1  RidgeClassifier using tvec Classification Report

#### 4.6.3.2  RidgeClassifier tvec Top 20 Features

#### 4.6.4.0 Random Forest using tvec 

#### 4.6.4.1 Random Forest using tvec Classification Report

#### 4.6.4.2 Random Forest tvec Top 20 Features

# 5.0 Modelling Conclusions

### 5.1 Model Performance
|Model|Train Score|Test Score|F1-Score|
|---|---|---|---|
|Naive Bayes with cvec|0.811|0.784|0.79|
|Naive Bayes with tvec|0.837|0.789|0.80|
|Random Forest with cvec|0.995|0.788|0.79|
|Random Forest with tvec|1.0|0.790|0.79|
|Logistic Regression with cvec|0.859|0.792|0.79|
|Logistic Regression with tvec|0.879|0.798|0.80|
|RidgeClassifier with cvec|0.914|0.786|0.79|
|RidgeClassifier with tvec|0.855|0.795|0.80|

All of our models perform similarly. However, the one can be further improve on is Naive Bayes CountVectorizer, the train and test score range where not far apart. 
 
The preferred production model is therefore Naive Bayes with CountVectorizer.

This model can be deployed in GPU discussion forums to separate posts about nVidia / AMD for sentiment analysis.

### 5.2 Lesson learn

Using NLTK to preprocess the text would product slightly higher score in terms of accuracy as compared using spaCY.

I had try to keep the model number and model name for all Nvidia graphic cards, In hope that it might helps the model to distinguish between Amd and Nvidia, but it didn't help the model to perform better either. 

I had tried to scrap all the comments from both subreddit amd and nvidia and merge it with the body text of the respective subreddit, after sending it to pre processsing using NLTK or spaCy, it does not improve the model performance. 

what work was instead of merging comments with body text, I used body text and merge with title and pre processing with NLTK, it will be able to produce the best score that I had so far. 

# 6.0 Sentiment Analysis Conclusion 

### Headline Conclusion Using VADER compound score 

1. The mean value of r/nvidia compound score was 0.207 

2. The mean value of r/amd compound score was 0.137 

3. Using the compound score, r/amd generally has a more negative sentiment as compare to r/nvidia

4. Hence, it is likely a genuine issue with AMD product 

# 7.0 Recommendations

## Modelling

1. The Naive Bayes with CountVectorizer.r model has the best result and run time, hence it can be deployed as to GPU discussion forums. 

2. While the F1-score % and accuracy around 79% it is good enough for our case. Our purpose is to roughly estimate whether there is any significantly more negative sentiment surround either AMD or Nvidia. The proportion of false positive and false negatives is decent based on the f1 score 


## Senitment Analysis 

1. Engineering needs to be given more resources to apply towards development for greater driver stability. 

2. A good start would be development and implementation of comprehensive testing suite before any new grpahics release. 

