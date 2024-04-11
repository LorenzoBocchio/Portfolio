---
layout: default
title:  Clustering Applied on Covid-19 themed tweets
description: Comparing various clustering techniques applied on tweets about Covid-19 in the United States
img: assets/img/covid_tweet.png
importance: 1
category: NLP Projects
---

## [Section 1: Preprocessing](#preproc)
## [Section 2: Vectorizing and Clustering](#vecclust)
## [Section 3: Conclusions ](#concl)


<b>Document clustering</b> is a field that lies at the intersection of Natural Language Processing (NLP) and Machine Learning (ML). Its primary purpose is to organize textual documents by grouping them into clusters based on their similarities and differences. This technique is used to discover hidden relationships and patterns within large datasets or document collections that may not be immediately apparent.


#  Section 1: Preprocessing
<a id='preproc'></a>

The dataset received is composed of information about tweets, it has almost a million rows and 13 columns.
These columns contains some details about the user who tweeted and about the tweet itself.
I'm going to focus only on the text of the tweets, utilizing a series of Natural Language Processing techniques for the preprocessing, a couple of vectorizing algorithms, and 5 clustering methods.


1. [Loading the dataset and libraries](#reading)
1. [Removing usernames](#usernames)
1. [Removing hashtags](#hashtags)
1. [Removing URLs](#url)
1. [Remobing  emojis](#emojis)
1. [Replacing underscores](#underscore)
1. [Replacing hyphens](#hyphen)
1. [Removing punctuations](#punctuations)
1. [Making the text lowercase](#lower_case)
1. [Language Detection](#langdetect)
1. [Removing Numbers](#numbers)
1. [Removing stopwords](#stopwords)
1. [Tokenization](#token)
1. [Stemming](#stemming) 
1. [WordCloud](#wordcloud)
1. [Training and Test](#traintest)


***
# 1.1: Load Data and Packages

<a id='reading'></a>
## 1.1: Importing packages and the dataset
Let's import the packages I'll use in this notebook, and a series of functions I built for the project.


```python
%run setup.ipynb 
```


Let's load the data from a csv file. Focusing only on the text of the tweets.



```python
df = pd.read_csv(f"{DATA_PATH}/Inhwa_tweets_covid19.csv")
df.shape
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>user_location</th>
      <th>text</th>
      <th>coordinates</th>
      <th>geo</th>
      <th>user_name</th>
      <th>user_created</th>
      <th>user_followers</th>
      <th>id_str</th>
      <th>created</th>
      <th>retweet_count</th>
      <th>polarity</th>
      <th>subjectivity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Republic of the Philippines</td>
      <td>RT @TheOnion: Xi Jinping Vows To Combat Corona...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Lolvoxell</td>
      <td>2018-08-29T13:27:53</td>
      <td>121</td>
      <td>1.221910e+18</td>
      <td>2020-01-27T21:31:40</td>
      <td>0</td>
      <td>-0.5000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Lagos - Ife</td>
      <td>RT @Leaux504: what the FUCK u mean they left?!...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>femi0la</td>
      <td>2017-11-04T15:55:23</td>
      <td>16453</td>
      <td>1.221910e+18</td>
      <td>2020-01-27T21:31:40</td>
      <td>0</td>
      <td>-0.2375</td>
      <td>0.429167</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Perú</td>
      <td>RT @ReutersLatam: La Bolsa de Nueva York cierr...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>VladiLarrea</td>
      <td>2010-01-19T19:35:06</td>
      <td>387</td>
      <td>1.221910e+18</td>
      <td>2020-01-27T21:31:40</td>
      <td>0</td>
      <td>0.0000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>NaN</td>
      <td>RT @TravelGov: #China Travel Advisory Update -...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>AuditionHengbok</td>
      <td>2009-06-19T08:28:07</td>
      <td>41</td>
      <td>1.221910e+18</td>
      <td>2020-01-27T21:31:40</td>
      <td>0</td>
      <td>0.0625</td>
      <td>0.354167</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>NaN</td>
      <td>RT @Leaux504: what the FUCK u mean they left?!...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MadisonMarq</td>
      <td>2015-06-26T00:51:08</td>
      <td>364</td>
      <td>1.221910e+18</td>
      <td>2020-01-27T21:31:40</td>
      <td>0</td>
      <td>-0.2375</td>
      <td>0.429167</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>The Philippines</td>
      <td>RT @eyekon131: A man from China is risking his...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>TakeTheTestProj</td>
      <td>2012-04-06T04:30:04</td>
      <td>320</td>
      <td>1.221910e+18</td>
      <td>2020-01-27T21:31:40</td>
      <td>0</td>
      <td>0.0000</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Lombardia, Italia</td>
      <td>RT @BarbaraRaval: In Austria e in Costa d'Avor...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MariangelaErcu3</td>
      <td>2019-02-07T15:40:47</td>
      <td>905</td>
      <td>1.221910e+18</td>
      <td>2020-01-27T21:31:40</td>
      <td>0</td>
      <td>-0.1250</td>
      <td>0.375000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Perchée tout là haut!</td>
      <td>Certains ont besoins de piment dans leur vie, ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>lavoixdenath</td>
      <td>2014-06-17T19:36:01</td>
      <td>117</td>
      <td>1.221910e+18</td>
      <td>2020-01-27T21:31:40</td>
      <td>0</td>
      <td>0.0000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>NaN</td>
      <td>RT @martiro10: Que mierda no van a tener el co...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>_vxckx_</td>
      <td>2018-05-10T16:40:54</td>
      <td>28</td>
      <td>1.221910e+18</td>
      <td>2020-01-27T21:31:40</td>
      <td>0</td>
      <td>0.0000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>La Joya De Los Sachas, Ecuador</td>
      <td>RT @LEXKANDER: 🇧🇪Plan contra corona virus\n🇧🇷P...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>DarwinChavezR2</td>
      <td>2015-12-12T21:16:10</td>
      <td>26</td>
      <td>1.221910e+18</td>
      <td>2020-01-27T21:31:40</td>
      <td>0</td>
      <td>0.0000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



Let's keep only the _text_ column. It's noticeable how tweets present usernames, hashtags, punctuation, emojis. Let's clean the texts, keeping only the ones written in English to help the clustering algorithms.


```python
df.dropna(subset = ["text"], inplace=True)
df = df[['text']]
df.shape
```




    (903922, 1)



***





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>80</th>
      <td>RT @FinnaganMarina: Black Friday Wuhan China S...</td>
    </tr>
    <tr>
      <th>81</th>
      <td>Muy preocupante...👇</td>
    </tr>
    <tr>
      <th>82</th>
      <td>RT @EpochTimes: #Virginia health officials sai...</td>
    </tr>
    <tr>
      <th>83</th>
      <td>@_neoparisien N'achète pas de masque\nÉduque l...</td>
    </tr>
    <tr>
      <th>84</th>
      <td>RT @jonoread: I swear this challenge is how co...</td>
    </tr>
    <tr>
      <th>85</th>
      <td>RT @samdastyari: DAILY FAIL: \n\nWhen our majo...</td>
    </tr>
    <tr>
      <th>86</th>
      <td>RT @PatriziaRametta: PRIMO CASO SOSPETTO IN AF...</td>
    </tr>
    <tr>
      <th>87</th>
      <td>RT @altNOAA: Airports are using thermal camera...</td>
    </tr>
    <tr>
      <th>88</th>
      <td>🤡🤡</td>
    </tr>
    <tr>
      <th>89</th>
      <td>RT @Interia_Fakty: Rząd Japonii zamierza samol...</td>
    </tr>
  </tbody>
</table>
</div>



<a id='usernames'></a>

## 1.2: Removing usernames
Let's remove the usernames and the retweeted usernames.


```python
remove_usernames = lambda x: re.sub('@([a-zA-Z0-9_]{1,50} +)', ' ',  x)
df.text = df.text.apply(remove_usernames)
```


```python
remove_usernamesRT = lambda x: re.sub('RT @([a-zA-Z0-9_]{1,50}: +)', ' ',  x)
df.text = df.text.apply(remove_usernamesRT)
```

<a id='hashtags'></a>

## 1.3: Removing hashtags
Next, all the hashtags symbols are removed, but I'm going to keep the "content" of the hashtag, since it's usually informative of the content and topic of the tweets.


```python
remove_hastags = lambda x: re.sub('#', ' ', x)
df.text = df.text.apply(remove_hastags)
```

<a id='url'></a>

## 1.4: Removing URLs
Now let's remove the URLs.


```python
remove_url = lambda x: re.sub(r'https\S+',' ', x)
df.text = df.text.apply(remove_url)
```

<a id='emojis'></a>

## 1.5: Remove emojis
Now onto the emojis. I'm going to use a function included in the setup notebook.


```python
df.text = df.text.apply(remove_emoji)
```

<a id='underscore'></a>

## 1.6: Replacing underscores and backslash

Some words can be separated by underscores. In order to continue the analysis, it's necessary to divide those words. 
Also, the new line symbol _\n_ can be found in some tweets. It can be removed.


```python
replace_underscore = lambda x: x.replace("_", " ")

replace_backslash = lambda x: x.replace("\n", " ")
```


```python
df.text = df.text.apply(replace_underscore)
df.text = df.text.apply(replace_backslash)
```

<a id='hyphen'></a>

## 1.7: Replacing hyphens

Replacing the ```-``` symbol can prevent some problems given by two words read as a single one.


```python
replace_hyphen = lambda x: x.replace("-", " ")
df.text = df.text.apply(replace_hyphen)
```

<a id='punctuations'></a>

## 1.8: Removing punctuations
The punctuation is removed.


```python
remove_punctuation = lambda x: re.sub(r'[^\w\s]',' ', x)
df.text = df.text.apply(remove_punctuation)
```

<a id='lower_case'></a>

## 1.9: Making lowercase
Let's make all the words lowercase, because Python is case sensitive. 


```python
to_lower = lambda x: x.lower()
df.text = df.text.apply(to_lower)
```



<a id='langdetect'></a>
## 1.10: Language Detection
Now that some of the most problematic content of the tweets has been removed, it's possible to apply the language detection of the text, so that the analysis can go on with only the English tweets.
The function  _detect_ is imported from the package ```langdetect```.

A function called _det_ is defined, so that, using try and except, makes the function keep working even if it cannot detect the language of a row. Rows with unclear language will be displayed as 'Other'.

The following operation will take at least 30 minutes, so it's advisable to run it only if needed.

```python
def det(x):
    try:
        lang = detect(x)
    except:
        lang = 'Other'
    return lang

df['language'] = df['text'].apply(det)
```

Since the operation above takes a while, I save the results in a new csv file. Let's load it.

```python
df.to_csv(f"{DATA_PATH}/dataLanguage.csv")
```

```python
dfLANG = pd.read_csv(f"{DATA_PATH}/dataLanguage.csv")
dfLANG.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>text</th>
      <th>language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>xi jinping vows to combat coronavirus by maki...</td>
      <td>en</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>what the fuck u mean they left</td>
      <td>en</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>la bolsa de nueva york cierra una jornada neg...</td>
      <td>es</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>hashtag china travel advisory update   level ...</td>
      <td>en</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>what the fuck u mean they left</td>
      <td>en</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>a man from china is risking his life to spill...</td>
      <td>en</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>in austria e in costa d avorio due casi sospe...</td>
      <td>it</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>certains ont besoins de piment dans leur vie  ...</td>
      <td>fr</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>que mierda no van a tener el coronavirus esto...</td>
      <td>es</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>plan contra corona virus plan contra corona v...</td>
      <td>es</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>quand coronavirus et lubrizol se croisent à r...</td>
      <td>fr</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>sees the news about the corona virus outbrea...</td>
      <td>en</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>bfmtv qui cherche un lien entre le coronaviru...</td>
      <td>fr</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>but as time pass  i am slowly realizing what ...</td>
      <td>en</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>now now they have come out with antivirus spra...</td>
      <td>en</td>
    </tr>
  </tbody>
</table>
</div>



The result seems satisfying! Even if the function could have misclassfied some English tweets (shorter texts, misspelled words and abbreviations play a role), there are still over half a million tweets for the analysis.
Let's just consider the English tweets.


```python
dfENG = dfLANG[dfLANG["language"] == "en"]
```


```python
dfENG.head(10), dfENG.shape
```




    (    Unnamed: 0                                               text language
     0            0   xi jinping vows to combat coronavirus by maki...       en
     1            1            what the fuck u mean they left                en
     3            3   hashtag china travel advisory update   level ...       en
     4            4            what the fuck u mean they left                en
     5            5   a man from china is risking his life to spill...       en
     11          11    sees the news about the corona virus outbrea...       en
     13          13   but as time pass  i am slowly realizing what ...       en
     14          14  now now they have come out with antivirus spra...       en
     16          16   san lazaro hospital  two of the three chinese...       en
     19          19  what are our nigerian medical scientists doing...       en,
     (541644, 3))



Let's continue with the text preprocessing.


<a id='numbers'></a>

## 1.11: Removing Numbers
Let's remove numbers from the text.


```python
remove_numbers = lambda x: re.sub(r'\d+', '', x)
dfENG.text = dfENG.text.apply(remove_numbers);
```


<a id='stopwords'></a>

## 1.12: Removing stopwords
Using the ```nltk``` package, each tweet is going to lose its stopwords. Let's  set the stopwords' language to English, and add a list of words that add no information to the analysis that can be found by looking at the data.


```python
stop = stopwords.words('english')
newStopwords = ["u", "ur", "u're", "hashtag", "amp", "one", "1", "2", "3", "4", "5", "6", "7", "8", "9", "000", 
               "like", "say", "says", "news", "get", "gets", "said", "go", "going", "york", "would", "well", "watch", "want",
               "video", "via", "use"]
```


```python
stop.extend(newStopwords)
```

Now I'm going to apply the function remove_stopwords (again, included in the setup), creating a new text variable "text_nostop" which cointains the tweets without the stopwords.


```python
dfENG["text_nostop"] = dfENG.text.apply(lambda text: remove_stopwords(text));
```

    

Let's see the most common words: the dataset theme is definitely the Coronavirus seen from an American point of view. A possible reason for this is that English was chosen as the only possible language of the tweets.


```python
cnt = Counter()
for text in dfENG["text_nostop"].values:
    for word in text.split():
        cnt[word] += 1
```


```python
cnt.most_common(50)
```




    [('coronavirus', 214298),
     ('covid', 82688),
     ('people', 40875),
     ('trump', 37734),
     ('virus', 33934),
     ('new', 31455),
     ('china', 30772),
     ('cases', 28634),
     ('us', 23880),
     ('corona', 22177),
     ('wuhan', 18908),
     ('health', 16959),
     ('pandemic', 16898),
     ('deaths', 15399),
     ('world', 14993),
     ('chinese', 13105),
     ('today', 12382),
     ('breaking', 12282),
     ('president', 12209),
     ('first', 11903),
     ('government', 11780),
     ('time', 11495),
     ('outbreak', 11044),
     ('death', 10673),
     ('state', 10438),
     ('day', 10431),
     ('spread', 10262),
     ('know', 10172),
     ('home', 9904),
     ('need', 9693),
     ('help', 9515),
     ('positive', 9485),
     ('many', 9281),
     ('testing', 9270),
     ('due', 8749),
     ('uk', 8719),
     ('lockdown', 8633),
     ('please', 8633),
     ('th', 8563),
     ('country', 8513),
     ('hospital', 8451),
     ('still', 8020),
     ('could', 8006),
     ('americans', 7919),
     ('may', 7802),
     ('take', 7790),
     ('died', 7755),
     ('right', 7629),
     ('patients', 7577),
     ('think', 7535)]



<a id='token'></a>

## 1.13: Tokenization
Let's divide each tweet into an array of words.


```python
dfENG['text_token']= dfENG['text_nostop'].apply(nltk.tokenize.WhitespaceTokenizer().tokenize);
```

    



<a id='stemming'></a>

## 1.14: Stemming
Let's transform each word to its root, using the defined function _stemming()_. Then, I tokenize these new stemmed tweets.


```python
dfENG["text_stemmed"] = dfENG["text_nostop"].apply(stemming)
```


```python
dfENG['text_stemmed_tokenized']= dfENG['text_stemmed'].apply(nltk.tokenize.WhitespaceTokenizer().tokenize)
```

Let's save this dataset.

```python
dfENG.to_csv(f"{DATA_PATH}/dataENGfinal.csv", index=False)
```

```python
dfENG =  pd.read_csv(f"{DATA_PATH}/dataENGfinal.csv")
```


```python
dfENG.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>language</th>
      <th>text_nostop</th>
      <th>text_token</th>
      <th>text_stemmed</th>
      <th>text_stemmed_tokenized</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>xi jinping vows to combat coronavirus by maki...</td>
      <td>en</td>
      <td>xi jinping vows combat coronavirus making ille...</td>
      <td>['xi', 'jinping', 'vows', 'combat', 'coronavir...</td>
      <td>xi jinp vow combat coronaviru make illeg menti...</td>
      <td>['xi', 'jinp', 'vow', 'combat', 'coronaviru', ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>what the fuck u mean they left</td>
      <td>en</td>
      <td>fuck mean left</td>
      <td>['fuck', 'mean', 'left']</td>
      <td>fuck mean left</td>
      <td>['fuck', 'mean', 'left']</td>
    </tr>
    <tr>
      <th>2</th>
      <td>hashtag china travel advisory update   level ...</td>
      <td>en</td>
      <td>china travel advisory update level reconsider ...</td>
      <td>['china', 'travel', 'advisory', 'update', 'lev...</td>
      <td>china travel advisori updat level reconsid tra...</td>
      <td>['china', 'travel', 'advisori', 'updat', 'leve...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>what the fuck u mean they left</td>
      <td>en</td>
      <td>fuck mean left</td>
      <td>['fuck', 'mean', 'left']</td>
      <td>fuck mean left</td>
      <td>['fuck', 'mean', 'left']</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a man from china is risking his life to spill...</td>
      <td>en</td>
      <td>man china risking life spill truth current sit...</td>
      <td>['man', 'china', 'risking', 'life', 'spill', '...</td>
      <td>man china risk life spill truth current situat...</td>
      <td>['man', 'china', 'risk', 'life', 'spill', 'tru...</td>
    </tr>
  </tbody>
</table>
</div>




<a id='wordcloud'></a>

## 1.15: WordCloud
Let's use WorldCloud to visualise the content of textual data in a quick and intuitive way. The idea behind WordClouds is based on showing the predominant words in a chunk of text with a positive correlation between the frequency and the font size.

So by using it I'm  going to see what the tweets are saying overall. 


```python
lists = dfENG['text_token'].tolist()
strings=' '.join(str(i) for i in lists)
```

Now let's  run the function WordCloud with the string above. I use 50 words so that the Wordcloud is clean and legible.


```python
wordcloud = WordCloud(width=1600, stopwords=stop,height=800,max_font_size=200,max_words=50,collocations=False, background_color='black').generate(strings)
plt.figure(figsize=(40,30))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
```


    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/project1/Preproc_WordCloud.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div> 
</div>

It is possible to see that the most common words are Coronavirus, Covid, people and Trump, since they are the words with a bigger font. This hints again on how the main topic of our database is related to COVID 19 in the US point of view.


<a id='traintest'></a>

## 1.16: Training and Test Split
Let's split the dataset into training and test data, so that it'll be possible to test the conclusions with a dataset which wasn't used for making assumptions. I am going to use 80% of the tweets in the training part.


```python
n_rows, n_cols = dfENG.shape
n_train = int(n_rows * 0.8)
train_index = choice(range(n_rows), n_train, replace=False)
test_index = [i for i in range(n_rows) if i not in train_index]
train_index = train_index.tolist()
dfENG_TRAIN = dfENG.iloc[train_index]
dfENG_TEST = dfENG.iloc[test_index]
```






It's now time to utilize vectorizing and clustering techniques.



#  Section 2: Vectorizing and Clustering
<a id='vecclust'></a>

1. [TfIdf Vectorizer](#tfidf)
1. [CountVectorizer](#count)
1. [Introduction to Clustering](#intro)
1. [K-Means](#kmeans)
1. [DBSCAN](#dbscan)
1. [Mini Batch K-Means](#mbk)
1. [BIRCH](#birch)
1. [Hierarchical Agglomerative Clustering](#hier)

After pre-processing the text data, let's generate some features.


Let's deal with NaN in the stemmed text, which can give problems with the vectorizing algorithms.


```python
dfENGnewTRAIN.text_stemmed.fillna(" ")
dfENGnewTEST.text_stemmed.fillna(" ")
dfENGnewTRAIN.text_stemmed = dfENGnewTRAIN.text_stemmed.astype(str)
dfENGnewTEST.text_stemmed = dfENGnewTEST.text_stemmed.astype(str)
```



## 2.1: TfIdf Vectorizer 
<a id='tfidf'></a>
 For document clustering, one of the most common ways to generate features for a document is to calculate the term frequencies of all its tokens, and sometimes it is also useful to weight the term frequencies by the inverse document frequencies. Although this method isn't flawless, it generally furnishes valuable insights into the document's primary topic. Moreover, applying weights to each term using its Inverse Document Frequency (IDF) across the document collection can enhance the feature set. This approach de-emphasizes common words, which typically hold little discriminatory power, thereby favoring less frequent but potentially more significant words.
Lastly, to account for documents of different lengths, each document vector is normalized so that it is of unit length.


```python
tfidf = TfidfVectorizer(
    min_df= 2500,
    max_df= 0.10, use_idf= True
)
tfidf.fit(dfENGnewTRAIN.text_stemmed)
text = tfidf.transform(dfENGnewTRAIN.text_stemmed)
```


Let's build the dataset obtained from the TFIDF vectorizer and have a look to it.


```python
tfidf_data = pd.DataFrame(text.toarray(), columns=tfidf.get_feature_names())
tfidf_data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>across</th>
      <th>actual</th>
      <th>administr</th>
      <th>ago</th>
      <th>allow</th>
      <th>alreadi</th>
      <th>also</th>
      <th>america</th>
      <th>american</th>
      <th>amid</th>
      <th>...</th>
      <th>wear</th>
      <th>week</th>
      <th>white</th>
      <th>without</th>
      <th>work</th>
      <th>worker</th>
      <th>world</th>
      <th>wuhan</th>
      <th>year</th>
      <th>yet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 254 columns</p>
</div>





## 2.2: CountVectorizer
<a id='count'></a>

CountVectorizer performs a simple word count, and returns a matrix where each row represents a document and each column represents a word. The entry at a specific row and column in the matrix represents the frequency of that word in the corresponding document.

While the CountVectorizer approach is simple and straightforward, it has some drawbacks. Let's see if the basic idea of CountVectorizer works, or if the better results are obtained with TFIDF.

```python
cv = CountVectorizer(min_df=2500, max_df= 0.1)
cv_matrix = cv.fit_transform(dfENGnewTRAIN.text_stemmed)
cv_matrix
cv_data = pd.DataFrame(cv_matrix.toarray(), columns=cv.get_feature_names())
```




## 2.3: Clustering Techniques
<a id='intro'></a>

In machine learning and statistics, clustering is a unsupervised learning task that is mainly used to reveal some hidden patterns that can be present in data but aren't immediately visible. 

The purpose of this method is the grouping of data based on common characteristics. This will be based on the similarity between instances measured by some distance metric, like the Euclidean distance. So instances in the same cluster must be as similar as possible, and data points of different clusters must be as much different as possible. 

There exists a variety of clustering algorithms, but not all are appropriate for every type of data. For example, the K-means method is only suitable for numerical data but requires the number of clusters prior to the construction of the algorithm while other type of algorithms like DBSCAN does not. 

Another important part of the clustering is the validation of the results, so determining the quality of the results obtained by the techniques is a key issue in this process. For the evaluation part one can use the Silhouette, the Davies-Bouldin or the Calinski-Harabasz index as an internal evaluation.

*Silhouette Index*: The silhouette score measures how similar a data point is to its own cluster compared to other clusters. The silhouette score ranges from -1 to 1. If the score is high, the data point fits well within the cluster. If it is low, the data point would fit better in another cluster. If the score is around 0, the data point could be assigned to any cluster. The silhouette score can provide an easily interpretable and visualizable measure of how well each data point fits within its cluster.

*Davies-Bouldin Index*: The Davies-Bouldin Index is another measure of the quality of a clustering algorithm. It uses a ratio of within-cluster distances to between-cluster distances. The Davies-Bouldin index values range from 0 to infinity. Lower values indicate better clustering because it means that the clusters are compact (smaller within-cluster distance) and well-separated (greater between-cluster distance).

*Calinski-Harabasz Index*: The Calinski-Harabasz Index, also known as the Variance Ratio Criterion, measures the dispersion within clusters and between clusters. Higher values of the index indicate better clustering. The index is the ratio of the sum of between-clusters dispersion and of within-cluster dispersion. In other words, clusters are dense and well separated if the index value is higher.

In this project I am working with five types of clustering algorithms: Mini Batch K-Means,DBSCAN, K-means, Hierarchical Agglomerative and BIRCH. To decide the appropriate number of clusters we utilize different methods: the elbow method with the values for the inertia and the distortion, and the 3 indexes referenced before. 



<a id='kmeans'></a>
## 2.4: K-Means
In this part we are going to apply the K-means clustering method.

This method starts with a pre-determined number of clusters. Then each observation is assigned to a cluster in order to minimize the within cluster sum of squares. The mean of the clustered observations is calculated and is then used as a new cluster centroid. After this step new observations are reassigned to the clusters and the centroids are recalculated repeteadly until the clusters don't change for each interation, so the algorithm converges.

The problem appears when there is a need to decide on a good and appropriate number of clusters.

To determine the number of clusters we will do a elbow graph and calculate the silhouette scores for the clusters.


There are two important ideas that I am going to use:

*Distortion*: it's the average of the squared distances from the cluster centers of the respective clusters. This distance can be calculated using the Euclidean distance metric.. 

*Inertia*: it's the sum of squared distances of the observations to their closest cluster center. 

So the next step is calculating the values of distortion and inertia for each value of K in the K-Means from a pre determined range. I will run several iterations, incrementing K and recording the SSE: 

Let's run the function using the tfidf_data dataset and using 24 as the maximum number of clusters. 


```python
%matplotlib inline 
find_optimal_clusters_Kmeans(tfidf_data, 24, f"{IMAGES_PATHtrain}/TRAIN_KMeans_tfidf_optimal.png" )
```  
    
<img src="../../assets/img/project1/output_32_1.png" width="765" height="505">




For this next step we are going to use the *Silhouette, Davies-Bouldin and Calinski-Harabasz indexes* to choose the optimal number of clusters. 

I created a function that gathers them all. 


```python
find_optimal_clustersKMeans_sil_DB_CH(tfidf_data, text, 24, 
                                      f"{IMAGES_PATHtrain}/TRAIN_KMeans_tfidf_optimal_silDBCH.png" )
```
    
<img src="../../assets/img/project1/output_34_1.png" width="765" height="505">

    


The optimal number of clusters seems to be 8. So, let's apply the K-Means to the tfidf dataset, considering 8 as number of clusters.


```python
clusters_tfidf_KM = KMeans(n_clusters=8, random_state=20).fit_predict(text)
```

The next step is to visualize the results by plotting the clusters using dimensionality reduction techniques. In this project I used the _t-SNE_ and the _PCA_ algorithm for this. 
 
The _t-SNE_ algorithm shows how the data is arranged in a low-dimensional space. It calculates a similarity measure between pairs of instances in the high dimensional space and in the low dimensional space. It then tries to optimize these two similarity measures. 

_PCA_ is a linear dimension reduction technique that seeks to maximize variance and preserves large pairwise distances. In other words, things that are different end up far apart. This can lead to poor visualization. 

_t-SNE_ differs from _PCA_ by preserving only small pairwise distances or local similarities, meaning that instances that are close to one another in the original high-dimensional dataset will remain close in the reduced-dimensional space. This makes it particularly suited for visualizing clusters in data.

In essence, t-SNE and PCA serve complementary roles in data visualization. While PCA helps in preserving global structure and can efficiently deal with a large number of dimensions, t-SNE excels in preserving local structure and is better suited for visualizing clusters. Therefore, using both in conjunction can provide comprehensive insights into the underlying structure of the data.


```python
plot_tsne_pca(text, clusters_tfidf_KM, f"{IMAGES_PATHtrain}/TRAIN_KMeans_tfidf_tsne_pca.png" )
```


 <img src="../../assets/img/project1/output_40_0.png" width="765" height="505">
   
    


The separation given by the t-sne is not precise, while the PCA plots show a decent clustering.
Let's plot in 3D the principal component. I don't focus on the first PC since it is usually a generic description of the variability of the data, and omitted plots confirm this.


```python
%matplotlib notebook
plot_3d_pca(text, clusters_tfidf_KM)
```

<table>
  <tr>
    <td><img src="../../assets/img/project1/clusters_tfidf_KM_output_angle_0.png" width="85%"></td>
    <td><img src="../../assets/img/project1/clusters_tfidf_KM_output_angle_90.png" width="85%"></td>
  </tr>
  <tr>
    <td><img src="../../assets/img/project1/clusters_tfidf_KM_output_angle_180.png" width="85%"></td>
    <td><img src="../../assets/img/project1/clusters_tfidf_KM_output_angle_270.png" width="85%"></td>
  </tr>
</table>

Here's the most frequent words for each cluster:

```python
topwords = get_top_keywords(text, clusters_tfidf_KM, tfidf.get_feature_names(), 10)
```

    
    Cluster 0
    mask,work,home,million,kill,mani,chines,infect,die,peopl
    
    Cluster 1
    peopl,us,new,american,respons,administr,call,donald,presid,trump
    
    Cluster 2
    week,state,patient,new,number,break,trump,peopl,posit,test
    
    Cluster 3
    first,break,state,number,total,death,report,confirm,new,case
    
    Cluster 4
    spread,travel,viru,chines,peopl,world,us,outbreak,wuhan,china
    
    Cluster 5
    govern,us,peopl,health,world,respons,amid,global,trump,pandem
    
    Cluster 6
    keep,chines,peopl,spread,call,shit,world,wuhan,corona,viru
    
    Cluster 7
    day,need,govern,spread,wuhan,health,new,time,death,us
    
Here is a possible interpretation of the content of each cluster:

**Cluster 0** is about Covid and infections;

**Cluster 1** deals with American politics;

**Cluster 2** seems about American politics and covid reports;

**Cluster 3** is about the daily report;

**Cluster 4** is about Covid in China;

**Cluster 5** is about US response to Covid;

**Cluster 6** could be about people reaction to the pandemic;

**Cluster 7** is about Covid spread and health measures;

`

Let's now try to apply K-Means to the __CountVectorizer__ data, and see if the fit is similar.


```python
%matplotlib inline 
find_optimal_clusters_Kmeans(cv_data, 24, f"{IMAGES_PATHtrain}/TRAIN_KMeans_cv_optimal.png" )
```



<img src="../../assets/img/project1/output_47_1.png" width="765" height="505">
    



```python
find_optimal_clustersKMeans_sil_DB_CH(cv_data, cv_matrix, 24, 
                                      f"{IMAGES_PATHtrain}/TRAIN_KMeans_cv_optimal_silDBCH.png" )
```

<img src="../../assets/img/project1/output_48_1.png" width="765" height="505">    
    


The optimal number of clusters seems to be 10.


```python
clusters_cv_KM = KMeans(n_clusters=10, random_state=20).fit_predict(cv_matrix)
```




```python
plot_tsne_pca(cv_matrix, clusters_cv_KM, f"{IMAGES_PATHtrain}/TRAIN_KMeans_cv_tsne_pca.png" )
```

<img src="../../assets/img/project1/output_52_0.png" width="765" height="505">
    


The clustering is not really clear: it's worse than the one obtained from the TFIDF method. The 3D plot improves slightly the division.


```python
plot_3d_pca(cv_matrix, clusters_cv_KM)
```

<table>
  <tr>
    <td><img src="../../assets/img/project1/clusters_cv_KM_output_angle_0.png" width="85%"></td>
    <td><img src="../../assets/img/project1/clusters_cv_KM_output_angle_90.png" width="85%"></td>
  </tr>
  <tr>
    <td><img src="../../assets/img/project1/clusters_cv_KM_output_angle_180.png" width="85%"></td>
    <td><img src="../../assets/img/project1/clusters_cv_KM_output_angle_270.png" width="85%"></td>
  </tr>
</table>


In order to understand the content of each group, let's have a look at the most frequent words for each group, and let's try to interpet the content. 


```python
topwords = get_top_keywords(text, clusters_cv_KM, cv.get_feature_names(), 10)
```

    
    Cluster 0
    need,hospit,peopl,patient,die,take,home,worker,health,care
    
    Cluster 1
    itali,rate,new,uk,report,number,case,us,toll,death
    
    Cluster 2
    need,week,number,state,patient,new,trump,peopl,posit,test
    
    Cluster 3
    world,health,need,govern,time,wuhan,new,pandem,us,china
    
    Cluster 4
    new,us,call,american,pandem,respons,administr,donald,presid,trump
    
    Cluster 5
    us,new,world,peopl,help,viru,prevent,china,stop,spread
    
    Cluster 6
    updat,day,state,number,death,total,report,confirm,new,case
    
    Cluster 7
    confirm,patient,state,test,report,china,first,case,trump,break
    
    Cluster 8
    shit,black,peopl,world,chines,china,wuhan,call,corona,viru
    
    Cluster 9
    work,wuhan,million,mani,kill,china,chines,infect,die,peopl
    

Here's the most frequent words for each cluster.

**Cluster 0** is about hospitals

**Cluster 1** deals with European reports of cases and deaths

**Cluster 2** seems about American politics and covid reports

**Cluster 3** is about Covid in general

**Cluster 4** is about US response to Covid
  
**Cluster 5** is about the spreading of Covid all over the world
  
**Cluster 6** is about the daily report
    
**Cluster 7** is about the first spread of the virus

**Cluster 8** is the people reaction to the virus

**Cluster 9** is about Covid in China



<a id='dbscan'></a>
## 2.5: DBSCAN

DBSCAN, short for Density-Based Spatial Clustering of Applications with Noise, is an algorithm designed to overcome certain limitations of other clustering methods such as K-Means and Hierarchical Clustering. Specifically, these methods tend to struggle with forming clusters of arbitrary shapes or different densities. DBSCAN, a density-based clustering algorithm, works on the premise that clusters are dense patches in the data space, segregated by regions of lower density.

DBSCAN's operation is based on just two parameters: epsilon and minPoints. Epsilon is the radius of the circle created around each data point to gauge the density.minPoints is the minimum number of data points required within the epsilon radius for the data point in question to be classified as a Core point. This algorithm also handles noise well; it labels any noisy samples as -1.

When it comes to selecting an appropriate value for minPoints, one common strategy is to use the natural logarithm of the total number of observations. To choose epsilon, I will run the algorithm making this parameter vary, and I'll choose the best one according to the number of noise observations and clusters.


```python
%matplotlib inline 
dbscanEPS(tfidf_data, 0.5, 1.6, f"{IMAGES_PATHtrain}/TRAIN_DBSCAN_tfidf_optimal.png")
```


<img src="../../assets/img/project1/output_62_1.png" width="765" height="505">

    


An eps value between 0.86 and 0.90 seems the best compromise between the number of clusters and the number of noise observations.


Let's apply the algorithm with epsilon equal to 0.90 to the TFIDF data, and with min_samples set to the logarithm of the number of observations.
```python
dbscan =  DBSCAN(eps=0.90, min_samples=math.log(tfidf_data.shape[0])).fit(tfidf_dataTRY)
clustersDBSCAN = dbscan.labels_
```


As the plots confirm, the clustering algorithm does not seem to be greatly useful in this analysis, since more than 85% of the observations lay in a single group.




```python
%matplotlib inline
plot_tsne_pca(tfidf_text, clustersDBSCAN, f"{IMAGES_PATHtrain}/TRAIN_DBSCAN_tfidf_tsne_pca.png")
```


<img src="../../assets/img/project1/output_68_0.png" width="765" height="505">

    



```python
#%matplotlib notebook
plot_3d_pca(tfidf_text, clustersDBSCAN)
```

<table>
  <tr>
    <td><img src="../../assets/img/project1/clustersDBSCAN_tfidf_output_angle_0.png" width="85%"></td>
    <td><img src="../../assets/img/project1/clustersDBSCAN_tfidf_output_angle_90.png" width="85%"></td>
  </tr>
  <tr>
    <td><img src="../../assets/img/project1/clustersDBSCAN_tfidf_output_angle_180.png" width="85%"></td>
    <td><img src="../../assets/img/project1/clustersDBSCAN_tfidf_output_angle_270.png" width="85%"></td>
  </tr>
</table>



Let's try with the __CountVectorizer__ dataset.



```python
%matplotlib inline
dbscanEPS(cv_data,1.5, 2.5, f"{IMAGES_PATHtrain}/TRAIN_DBSCAN_cv_optimal.png")
```

 <img src="../../assets/img/project1/output_75_1.png" width="765" height="505">
   


Looking at the 2 plots, an eps equal to 2.3 has a low number of noise observations and a number of clusters similar to what we observed with the other clustering methods.


```python
dbscancv =  DBSCAN(eps=2.3, min_samples=math.log(cv_data.shape[0])).fit(cv_data)
clustersDBSCANcv = dbscancv.labels_
```

    
Again, DBSCAN puts almost all the observation in one cluster, making the division useless.



```python
%matplotlib inline
plot_tsne_pca(cv_matrix, clustersDBSCANcv, f"{IMAGES_PATHtrain}/TRAIN_DBSCAN_cv_tsne_pca.png")
```

<img src="../../assets/img/project1/output_81_0.png" width="765" height="505">

    
    



```python
#%matplotlib notebook
plot_3d_pca(cv_matrixTRY, clustersDBSCANcv)
```


<table>
  <tr>
    <td><img src="../../assets/img/project1/clustersDBSCANcv_output_angle_0.png" width="85%"></td>
    <td><img src="../../assets/img/project1/clustersDBSCANcv_output_angle_90.png" width="85%"></td>
  </tr>
  <tr>
    <td><img src="../../assets/img/project1/clustersDBSCANcv_output_angle_180.png" width="85%"></td>
    <td><img src="../../assets/img/project1/clustersDBSCANcv_output_angle_270.png" width="85%"></td>
  </tr>
</table>




<a id='mbk'></a>
## 2.6: MiniBatchKMeans
The MiniBatchKMeans is a variant of the KMeans algorithm which uses mini-batches to reduce the computation time. Mini-batches are subsets of the input data, sampled randomly in each training iteration, so that the computation time for convergence is reduced, with only a slighlty worse quality of the clustering.

Similar to the classic K-Means algorithm, in the first step, samples are drawn randomly from the dataset, to form a mini-batch. These are then assigned to the nearest centroid. In the second step, the centroids are updated. In contrast to k-means, this is done on a per-sample basis. For each sample in the mini-batch, the assigned centroid is updated by taking the streaming average of the sample and all previous samples assigned to that centroid. This has the effect of decreasing the rate of change for a centroid over time. These steps are performed until convergence or a predetermined number of iterations is reached.

MiniBatchKMeans converges faster than KMeans, but the quality of the results is reduced. In practice this difference in quality can be quite small, as shown in the example.

Let's create a function for the application of the MiniBatchKmeans to the data.
The parameter batch_size is set to 2048, that is 256 multiplied by the number of cores for a faster result. Init-size is the number of samples used in the clustering.


Let's first apply MiniBatchKMeans to the TFIDF data.
```python
%matplotlib inline
find_optimal_clustersMBK(tfidf_data, 24,f"{IMAGES_PATHtrain}/TRAIN_MiniBatch_tfidf_optimal.png")
```
<img src="../../assets/img/project1/output_88_1.png" width="765" height="505">




```python
find_optimal_clustersMBK_sil_DB_CH(tfidf_data, text, 24, 
                                   f"{IMAGES_PATHtrain}/TRAIN_MiniBatch_tfidf_optimal_silDBCH.png")
``` 

<img src="../../assets/img/project1/output_90_1.png" width="765" height="505">       


When plotting SSE as a function of the number of clusters, it is possible to notice that it continues to decrease as k increases. As more centroids are added, the distance from each point to its closest centroid will decrease.

Sometimes the number of the clusters might not be straightforward. In that case we can use the library *kneed* to identify the elbow point. 

Using the elbow rule applied to inertia and distortion, 8 seems the ideal number of clusters.


```python
clustersTFIDF_MINIBATCHKM = MiniBatchKMeans(n_clusters=8, init_size=1024, batch_size=2048, random_state=20).fit_predict(text)
```

Let's see the t-sne and PCA plots, in order to understand if the clusters are separated in the space projected by the two methods.



```python
plot_tsne_pca(text, clustersTFIDF_MINIBATCHKM, f"{IMAGES_PATHtrain}/TRAIN_MiniBatch_tfidf_tsne_pca.png")
```


<img src="../../assets/img/project1/output_98_0.png" width="765" height="505">      


The cluster separation in the t-sne is not really clear, while the PCA plots show a good differentiation between the different 8 clusters. Let's see a 3D plot of the PCA.


```python
plot_3d_pca(text, clustersTFIDF_MINIBATCHKM)
```

<table>
  <tr>
    <td><img src="../../assets/img/project1/clustersMiniBatch_tfidf_output_angle_0.png" width="85%"></td>
    <td><img src="../../assets/img/project1/clustersMiniBatch_tfidf_output_angle_90.png" width="85%"></td>
  </tr>
  <tr>
    <td><img src="../../assets/img/project1/clustersMiniBatch_tfidf_output_angle_180.png" width="85%"></td>
    <td><img src="../../assets/img/project1/clustersMiniBatch_tfidf_output_angle_270.png" width="85%"></td>
  </tr>
</table>




The separation is quite clear. K-Means, even when applied using mini-batches, seems very useful in the separation in clusters of data built with TFIDF.

Here's the most frequent words for each cluster:

```python
topwords = get_top_keywords(text, clusters, tfidf.get_feature_names(), 10)
```

    
    Cluster 0
    new,spread,need,govern,health,time,us,pandem,test,peopl
    
    Cluster 1
    keep,chines,spread,shit,call,world,peopl,wuhan,corona,viru
    
    Cluster 2
    day,state,toll,number,total,report,confirm,new,death,case
    
    Cluster 3
    spread,travel,viru,chines,us,world,peopl,outbreak,wuhan,china
    
    Cluster 4
    nurs,home,care,doctor,recov,new,test,treat,hospit,patient
    
    Cluster 5
    test,man,ago,american,last,peopl,old,week,year,die
    
    Cluster 6
    peopl,american,pandem,respons,test,administr,call,donald,presid,trump
    
    Cluster 7
    us,make,safe,work,order,peopl,nurs,stay,look,home
    


**Cluster 0** is about healthcare and it's dealing with the pandemic

**Cluster 1** deals with the first people reactions of the discovery of covid in Wuhan

**Cluster 2** seems about the daily report of cases and death

**Cluster 3** seems to contain the first news about the virus

**Cluster 4** is about hospitals

**Cluster 5** is not really clear

**Cluster 6** is about the US response to the pandemic

**Cluster 8** is about safety



 Let's use __MiniBatchKMeans__ with the __CountVectorizer__ data.


```python
%matplotlib inline
find_optimal_clustersMBK(cv_data, 24, f"{IMAGES_PATHtrain}/TRAIN_MiniBatch_cv_optimal.png")
```

    
<img src="../../assets/img/project1/output_106_1.png" width="765" height="505">       
    



```python
find_optimal_clustersMBK_sil_DB_CH(cv_data, cv_matrix, 24, 
                                   f"{IMAGES_PATHtrain}/TRAIN_MiniBatch_cv_optimal_silDBCH.png" )
```


<img src="../../assets/img/project1/output_107_1.png" width="765" height="505">       

    
    


The optimal numbers of clusters according to the elbow method is 6, confirmed also by the three indexes.


```python
clusters_cv_minibatch = MiniBatchKMeans(n_clusters=6, init_size=1024, batch_size=2048, 
                           random_state=20).fit_predict(cv_matrix)
```



```python
plot_tsne_pca(cv_matrix, clusters_cv_minibatch, f"{IMAGES_PATHtrain}/TRAIN_MiniBatch_cv_tsne_pca.png" )
```


<img src="../../assets/img/project1/output_112_0.png" width="765" height="505">       
  



```python
plot_3d_pca(cv_matrix, clusters_cv_minibatch)
```

<table>
  <tr>
    <td><img src="../../assets/img/project1/clusters_cv_minibatch_output_angle_0.png" width="85%"></td>
    <td><img src="../../assets/img/project1/clusters_cv_minibatch_output_angle_90.png" width="85%"></td>
  </tr>
  <tr>
    <td><img src="../../assets/img/project1/clusters_cv_minibatch_output_angle_180.png" width="85%"></td>
    <td><img src="../../assets/img/project1/clusters_cv_minibatch_output_angle_270.png" width="85%"></td>
  </tr>
</table>





Here's the most frequent words for each cluster:

```python
topwords = get_top_keywords(text, clusters_cv, cv.get_feature_names(), 10)
```

    
    Cluster 0
    shit,black,trump,spread,world,chines,peopl,call,corona,viru
    
    Cluster 1
    spread,travel,viru,chines,world,us,peopl,outbreak,wuhan,china
    
    Cluster 2
    first,break,death,state,number,total,report,confirm,new,case
    
    Cluster 3
    need,die,health,time,us,new,pandem,test,peopl,trump
    
    Cluster 4
    state,new,rate,uk,report,number,case,us,toll,death
    
    Cluster 5
    quarantin,citi,doctor,mask,outbreak,peopl,chines,hospit,viru,wuhan
    


**Cluster 0** is a mixture of reaction about Covid and the protests in the US

**Cluster 1** deals with the first news about the discovery of the virus in Wuhan
  
**Cluster 2** seems about the daily report of cases and death

**Cluster 3** is about hospitals
    
**Cluster 4** is about the daily report
    
**Cluster 5** is about safety and hospitals


<a id='birch'></a>
## 2.7: BIRCH

Let's discuss the Birch (balanced iterative reducing and clustering using hierarchies) clustering method. This model involves the construction of a tree-like structure composed of nodes, known as Cluster Features (CF), which summarize the data. When a new data point is introduced to the model, it navigates down the tree to the CF leaf that it aligns with most closely. If the data point fits within the leaf and the leaf is not exceeding its capacity, the CF statistics are augmented for all nodes from the leaf right up to the root. In contrast, if the data point doesn't integrate well with the leaf or if the leaf is already at full capacity, a fresh CF is formed. Since the maximum number of children a node (branching factor) can have is capped, a split can occur one or more times. Once the tree reaches the allocated memory size, it undergoes reconstruction and the threshold – which decides whether a new point is assigned to a leaf or creates a new leaf – gets updated. Any outliers detected are refitted during subsequent tree rebuilds.

Let's first apply the Birch method to our TFIDF dataset.



I run the function using 24 as parameter for the number of maximum clusters. Observing the three indexes, the ideal number of clusters seems to be either 4, or around 14. I picked the latter, based on the fact that choosing 4 would lead to a cluster having the majority of the observations.


```python
find_optimal_clusters_birch(tfidf_data, text, 24, f"{IMAGES_PATHtrain}/TRAIN_BIRCH_tfidf_optimal.png")
```

<img src="../../assets/img/project1/output_122_1.png" width="765" height="505">       



Now, let's create the BIRCH clustering model and fitting the data with 14 clusters.


```python
birchPred=Birch(branching_factor=100,n_clusters=14,threshold=0.5).fit_predict(tfidf_data)
```

Let's plot the data using the functions defined before:



```python
%matplotlib inline 
plot_tsne_pca(text, birchPred, f"{IMAGES_PATHtrain}/TRAIN_BIRCH_tfidf_tsne_pca.png" )
```



<img src="../../assets/img/project1/TRAIN_BIRCH_tfidf_tsne_pca.png" width="765" height="505">       

    
  
There still is one cluster that has the majority of the observations in one cluster. The fact that the clusters are also overlapped shows how it is not easy to find dissimilarities in a TFIDF-generated matrix using BIRCH.

Let's plot in 3D. Even by plotting in three dimensions, the clusters are overlapped.


```python
%matplotlib notebook
plot_3d_pca(text, birchPred)
```

<table>
  <tr>
    <td><img src="../../assets/img/project1/Birch_tfidf_output_angle_0.png" width="85%"></td>
    <td><img src="../../assets/img/project1/Birch_tfidf_output_angle_90.png" width="85%"></td>
  </tr>
  <tr>
    <td><img src="../../assets/img/project1/Birch_tfidf_output_angle_180.png" width="85%"></td>
    <td><img src="../../assets/img/project1/Birch_tfidf_output_angle_270.png" width="85%"></td>
  </tr>
</table>



Since the results of the BIRCH with the __CountVectorizer__ data are even worse, I am going to omit them. 


<a id='hier'></a>
## 2.8: Hierarchical Agglomerative Clustering

This algorithm employs a hierarchical clustering approach. Initially, it treats each data point as an individual cluster and then merges them based on similarity. However, it requires O(n²) memory and O(n³) runtime, which presents challenges with high-dimensional data. The problem lies not in the vectorization parameters, but rather in the number of rows. Therefore, using more restrictive parameter values in the vectorizing algorithms is ineffective.

Let's decide the optimal number of clusters based on the 3 different indexes. Let's start by applying the clustering to the TFIDF data. 


```python
%matplotlib inline 
find_optimal_clusters_Hierarc(tfidf_data, text, 24,
                             f"{IMAGES_PATHtrain}/TRAIN_HierAgg_tfidf_optimal_silDBCH.png")
```

    
<img src="../../assets/img/project1/output_154_1.png" width="765" height="505">       



Choosing the number of clusters here is a bit difficult: the silhouette score is low for every possible number, given the fact that the silhouette score is between -1 and 1. The maximum is with the lowest possibility (2), but that seems quite useless in terms of clustering results.
I am then going to utilize 10 as number of clusters, since it lead to optimal results with previous clustering techniques. 


```python
clusterHierarc = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')
clusterHierarc_TFIDF = clusterHierarc.fit_predict(tfidf_data)
```


The clustering leads to one group containing the majority (more than 85%) of the observations.




```python
%matplotlib inline 
plot_tsne_pca(text, clusterHierarc_TFIDF, f"{IMAGES_PATHtrain}/TRAIN_HierAgg_tfidf_tsne_pca.png")
```

<img src="../../assets/img/project1/TRAIN_HierAgg_tfidf_tsne_pca.png" width="765" height="505">       

  



```python
#%matplotlib notebook
plot_3d_pca(textREDUCED, clusterHierarc_TFIDF)
```

<table>
  <tr>
    <td><img src="../../assets/img/project1/clusterHierarc_TFIDF_output_angle_0.png" width="85%"></td>
    <td><img src="../../assets/img/project1/clusterHierarc_TFIDF_output_angle_90.png" width="85%"></td>
  </tr>
  <tr>
    <td><img src="../../assets/img/project1/clusterHierarc_TFIDF_output_angle_180.png" width="85%"></td>
    <td><img src="../../assets/img/project1/clusterHierarc_TFIDF_output_angle_270.png" width="85%"></td>
  </tr>
</table>

The groups are divided in an acceptable way by the use of principal components.

Since the results are not great, I am not going to include the output I got from the hierarchical clustering applied to the CV data.




#  Section 3: Conclusions
<a id='concl'></a>

It's now time to test the results obtained in the clustering project: am going to apply the same tecniques but with the knowledge gained from the training step, such as the more suitable numbers of clusters for each of the algorithms we apply to the data. The test set is composed of 20% of the original dataset, with 108328 chosen randomly from the over 500.000 observations obtained from keeping only the English written tweets from the original dataset.

In the training set, all five clustering algorithms produced clusters with low Silhouette scores. This may be attributed to the nature of the datasets: both are sparse and have non-null values for a minimal fraction of columns.

Comparing the clusterings from the test set, the differences between TF-IDF and CountVectorizer are not significant. Although TF-IDF may yield marginally better results due to its term frequency-inverse document frequency scores, its performance does not vastly surpass CountVectorizer, which uses simpler absolute frequencies.

Among the five clustering algorithms, K-Means and Mini Batch K-Means significantly outperformed the others:

- Their clustering results are more discernible in t-SNE and PCA plots, indicating that the identified clusters possess distinct characteristics reflecting data variability.

- They differentiate clusters based on the most frequent words, indicating different topics for almost every group. These topics are subjectively chosen, without statistical validity, but provide useful insights.

- They are faster, especially Mini Batch K-Means, due to its use of batches.

In contrast, DBSCAN, BIRCH, and Hierarchical Agglomerative Clustering faced limitations:

- They struggle with large datasets, and even when trying to use a subset of the data, these algorithms are imprecise, with overlapping groups in the t-SNE and PCA plots.


In conclusion, K-Means and Mini Batch K-Means, alongside the vectorizing algorithms, provide useful techniques for text datasets. They should be considered for future NLP problems, though they require intensive preprocessing to effectively leverage the data's most informative aspects.
