---
layout: default
title: "Sentiment and Semantics: Applying NLP to Uncover Media Framing Techniques in Environmental News"
description: Analyzing and comparing the framing of environmental news by Fox News and National Public Radio (NPR) using NLP

img: assets/img/planet.jpeg
importance: 1
category: NLP Projects
---
  
#### [Section 1: Introduction](#intro)
#### [Section 2: Methodology](#method)

###### [2.1: Data Scraping](#scrape)

###### [2.2: Preprocessing](#Preprocessing)

###### [2.3: Exploratory Analysis](#expl)

###### [2.4: Sentiment Analysis](#sent)

###### [2.5: Topic Modeling](#topic)

#### [Section 3: Results ](#results)

###### [3.1: WordClouds for the most mentioned people](#wc)

###### [3.2: Sentiment scores for most mentioned people and topics](#sent)

###### [3.3: Applying Topic Modeling](#topicappl)

#### [Section 4: Conclusions](#concl)



***
## Section 1: Introduction

<a id='intro'></a>

The project tries to investigate the framing of environmental issues by two US media outlets - Fox News and National Public Radio (NPR). The choice of these two outlets was made because of their representation of opposite ideological viewpoints and their appeal to diverse audience demographics, which might lead to the detection of differences in their approach.

The study is based on a data-driven methodology that focuses on the application of Natural Language Processing (NLP) techniques. It includes the usage of WordClouds, Sentiment Analysis, and Topic Modeling to identify how the media outlets try to influence public perception with the use of linguistic framing in their news.


Environmental news is an increasingly crucial domain of journalism. The urgency and complexity of environmental issues demand careful attention from news organizations. Despite the growing importance of environmental news, existing research suggests that media outlets frame these topics differently, leading to varied public interpretations and responses.

## Section 2: Methodology

<a id='method'></a>



#### 2.1: Data Scraping 

<a id='scrape'></a>

The data were collected using web scraping techniques, obtaining all the news found in the following two categories of news archive from the very first one to the most recent at the time of the scraping (21/4/2023): [Fox News](https://www.foxnews.com/category/us/environment/climate-change) and  [NPR](https://www.npr.org/sections/environment/).

With this method, the dataset is composed of 1132 Fox News articles ranging from 2015 to 2023, and 1740 NPR articles,from 2017 to 2023. Every single article includes three different variables: a tag that classifies the news, the headline and the lead.



#### 2.2: Preprocessing 

<a id='Preprocessing'></a>

The very first procedure needed to apply natural language processing (NLP) techniques is the preprocessing. Preprocessing aims to prepare the data for the subsequent statistical analysis. Usernames, hashtags, URLs, emojis, punctuation, numbers are removed from the headlines and the leads. Underscores, backslashes, and hyphens are replaced with a word space. Headlines and leads are transformed into lowercase characters, and stopwords are removed from these column. The stopwords dictionary is taken from the Python library <i>NLTK</i>, to which was added a new list of words specific of the scraped data. This added words don't include actual useful information, such as "download" or "transcript".

The next operation is tokenization. Tokenization is a procedure that divides a string of text into an array of single words. It is applied to headlines and leads.



#### 2.3: Exploratory Analysis 

<a id='expl'></a>
When dealing with textual data, a first look at the content of the news can be obtained by plotting WordClouds and by focusing on the most frequent bigrams and trigrams. The idea behind WordClouds is based on showing the predominant words in a chunk of text with a positive correlation between the frequency and the font size. Bigrams and trigrams are respectively sequences of two and three consecutive words that appear together in a specific order within a sentence. 

<table>
<caption> Figure 1: WordClouds of Fox News and NPR</caption>
  <tr>
    <td><img src="../../assets/img/project_env/1.png" width="155%"></td>
    <td><img src="../../assets/img/project_env/2.png" width="155%"></td>
  </tr>
</table>

Examining the two WordClouds, it is complex to find specific differences, since the content of the scraped news is about climate change, meaning that the most frequent words are going to be similar: for both Fox News and NPR, words that stand out are "climate", "change", "new", "California", "Biden".


<table>
<caption>Figure 2: Bigrams and Trigrams of Fox News and NPR</caption>
  <tr>
    <td><img src="../../assets/img/project_env/bi_tri_foxNews.png" width="155%"></td>
    <td><img src="../../assets/img/project_env/bi_tri_NPR.png" width="155%"></td>
  </tr>
</table>



The tables in Figure 2 show the top 20 bigrams and trigrams for Fox News and NPR, sought in a new variable containing the union of headline and lead for each article. The reason for the use of this new variable is found in the will to focus on as many words as possible for every news. Although similarities are still present, it is possible to affirm that Fox News seems to include people's name more frequently: US Presidents Biden and Trump, politicians such as John Kerry, Alexandria Ocasio-Cortez, Gavin Newsom, Joe Manchin, political commentator Tucker Carlson, and activist Greta Thunberg are present in the top 20. The same cannot be said for NPR, where the only individuals present in the bigrams and trigrams are US Presidents Biden and Trump, and reporters Lauren Sommer and Rebecca Hersher.

Moving the focus on the most frequent topics mentioned, Fox News has a big emphasis on the proposed but yet to be implemented "Green New Deal", with it being the most frequent trigram, while NPR does not have it in its top 20. 

"Climate Change", "Fossil fuel", and "Global warming" appear in both newspapers' rankings. It is possible to see how NPR also focuses specifically on topics associated with climate change, such as "Renewable energy", "Greenhouse gas", and "Carbon emissions".




#### 2.4: Sentiment Analysis 

<a id='sent'></a>


Sentiment Analysis is a technique that aims to systematically identify and quantify textual information. Here, sentiment analysis is applied to determine the tone and emotional content of news headlines, with the goal being understanding how the news outlets considered portray environmental issues. 
Sentiment analysis works by essentially assigning scores to words based on their semantic meaning. These scores range from negative to positive on a scale in the continuous interval [-1,1] and represent the sentiment of the word. Sentences, or in this case, news headlines and leads are then scored by computing the means of the scores of the words of the text.

In this specific project, the Valence Aware Dictionary and Sentiment Reasoner sentiment analysis tool (VADER) is used. VADER is a lexicon and rule-based sentiment analysis tool that was specifically created to score sentiments expressed in social media, or for a short text format.  VADER uses a combination of qualitative and quantitative techniques to assign scores to specific words according to the perceived sentiment and to the context.
Since VADER is able to handle scenarios that present negations (for example, "not good"), that can flip the sentiment of a phrase, the sentiment analysis is going to be applied on the union of headline and lead, without the removal of stopwords.

The aim is to understand whether environmental issues are consistently dramatized, and if there's a disparity between how these two outlets report on the same environmental topics. This could reveal potential bias in reporting or highlight different approaches to environmental journalism between the two.




Topic modeling is a method for unsupervised classification of documents, similar to clustering on numeric data, which recognizes natural hidden groups by identifying clusters of similar words.

The specific variant of topic modeling used in this project is called Latent Dirichlet Allocation (LDA). LDA is based on the assumption that each document is a mixture of various topics, and that every word within the document can be attributed to one of those topics. While these topics are not assigned labels, a close examination of the words most strongly associated with each one can offer interpretive insights into the semantic content of the topic.

LDA operates under a probabilistic framework involving a two-stage generative process for text. In the first stage, for each document *i* in *i = 1, ..., N*, (with *N* being the number of documents in the corpus) a topic distribution is sampled from a Dirichlet distribution, where *θ<sub>i</sub>* is sampled from Dir(α), which requires as input α, a K-dimensional vector, with K denoting the total number of topics.

In the second stage, for each of the K topics, a word distribution is sampled from another Dirichlet distribution, where *φ<sub>k</sub>* is sampled from Dir(β), requiring β, a V-dimensional vector, with V representing the size of the vocabulary, that is the list of all unique words found in all the documents.

Going into details, for the texts building operation, a topic is sampled from a multinomial distribution, and then a word is chosen based on this distribution. This process involves sampling a topic *z<sub>ij</sub>* from a multinomial distribution Multinomial(θ<sub>i</sub>), and then a word *w<sub>ij</sub>* is sampled from Multinomial(φ<sub>z<sub>ij</sub></sub>).

This two-stage process is iteratively conducted until the entire corpus has been generated, providing a topic mixture for each document and a word distribution for each topic.

The implementation of LDA used in this research comes from the Scikit-learn library. The model needs the input in the form of a bag-of-words model, which is a representation of text that describes the occurrence of words within a document.

To choose the optimal number of topics, two measures have been used: perplexity and coherence scores. Perplexity is a statistical metric utilized to determine how well a probabilistic model predicts a sample, and it is typically used to compare different probabilistic models. In the context of LDA, a lower perplexity score suggests superior generalization. Coherence score measures the degree of semantic similarity, thus providing an understanding of the topic's interpretability: a higher coherence score indicates that the topic's words collectively make more sense.

Since a correct clustering of texts does not exist per se, and topic modeling is merely an interpretation study, the final choice of the number of topics is also based on the interpretability of the groups that LDA creates. This means that having a low number of clusters would be too general, and, similarly, a high number of clusters would overfit. The number of topics that was decided for each outlet is 10.

## Section 3: Results

<a id='results'></a>



#### 3.1: WordClouds for the most mentioned people

<a id='wc'></a>

<table>
<caption>Figure 3: Fox Wordclouds</caption>
  <tr>
    <td><img src="../../assets/img/project_env/biden_fox_wordcloud.png" alt="Biden Fox Wordcloud" style="width: 100%;"/></td>
    <td><img src="../../assets/img/project_env/trump_fox_wordcloud.png" alt="Trump Fox Wordcloud" style="width: 100%;"/></td>
    <td><img src="../../assets/img/project_env/aoc_fox_wordcloud.png" alt="AOC Fox Wordcloud" style="width: 100%;"/></td>
  </tr>
  <tr>
    <td><img src="../../assets/img/project_env/kerry_fox_wordcloud.png" alt="Kerry Fox Wordcloud" style="width: 100%;"/></td>
    <td><img src="../../assets/img/project_env/thunberg_fox_wordcloud.png" alt="Thunberg Fox Wordcloud" style="width: 100%;"/></td>
  </tr>
</table>



<table>
<caption>Figure 4: NPR Wordclouds</caption>
  <tr>
    <td><img src="../../assets/img/project_env/biden_npr_wordcloud.png" alt="Biden NPR Wordcloud" style="width: 100%;"/></td>
    <td><img src="../../assets/img/project_env/trump_npr_wordcloud.png" alt="Trump NPR Wordcloud" style="width: 100%;"/></td>
    <td><img src="../../assets/img/project_env/kerry_fox_wordcloud.png" alt="Kerry NPR Wordcloud" style="width: 100%;"/></td>
  </tr>
</table>

Focusing on the WordClouds generated for the most mentioned individuals in the data of Fox News and NPR does not seem to show clear differences in the words used by the outlets to describe the same people. In particular, every plot presents words related to the environment and climate change. More interesting facts can be found focusing on the less frequent words (such "Private Jet" for the Fox News WordCloud for John Kerry), but this only provides a generic glimpse of the main concepts connected to every individual. This type of analysis is superficial and does not bring a quantifiable way to express how the media outlets describe the most mentioned people. This limitation is overcome in the following subsection.



#### 3.2: Sentiment scores for most mentioned people and topics


<a id='sent'></a>


Since journalistic narratives shape public understanding and engagement with environmental issues, it is crucial to understand how media frames work. As Walter Lippmann highlighted, "the real environment is altogether too big, too complex, and too fleeting for direct acquaintance". For this reason, media outlets build maps of the world to communicate stories, framing news in a manner that allows audiences to make sense of complex phenomena.

The sentiment analysis applied on news headlines and leads from Fox News and NPR offers interesting insight into the framing mechanisms employed by them. Specifically, it shows how the same topic can be represented differently depending on the context and the outlet's overall narrative framework.

Starting with the portrayal of political figures, Fox News appears to maintain a neutral stance towards both Presidents Biden and Trump, based on their sentiment scores. In contrast, NPR exhibits a slightly more positive representation of President Biden, with a sentiment score of 0.08.

As for the U.S. Special Presidential Envoy for Climate John Kerry, both media outlets display a similar sentiment positive score. 

The sentiment scores associated with US Representative Ocasio-Cortez and activist Greta Thunberg in Fox News are of particular interest. Ocasio-Cortez receives the lowest score of -0.18. On the other hand, Thunberg is represented with the highest score of 0.12 among all individuals rated in Fox News. The NPR scores are not presented, since the sample sizes of news mentioning the two are very low.



<figure>
 <img src="../../assets/img/project_env/finalSentim.png">
    <figcaption>Figure 5: Sentiment Scores for the most frequent themes and people in the news </figcaption>
</figure>

When considering the most frequently mentioned environmental topics, both outlets present the highest positive sentiment towards "Renewable Energy" (0.28 and 0.36 for Fox News and NPR, respectively) and a negative sentiment towards "Global Warming". These sentiment scores align with the broader global consensus that promotes renewable energy as a sustainable solution and perceives global warming as a significant threat. 

However, when the focus shifts to different environmental topics, from the general "Climate Change", to the more specific "Fossil Fuel", "Greenhouse Gas", and "Carbon Emissions", the sentiment scores highlight big differences between Fox News and NPR. Fox News exhibits very negative sentiment scores for these topics, possibly suggesting a dramatic or crisis-oriented frame. This form of media framing, often associated with the concept of "shrinking news hole", reflects the media's inclination to dramatize issues and prioritize attention-grabbing headlines to compete for audience attention.

On the other hand, NPR assigns more neutral to positive sentiment scores to these topics, presenting a more balanced and possibly constructive framing. This might be part of NPR's effort to portray environmental issues in a comprehensive and detailed manner.
 
Even if it is not always possible to affirm that media frames are able to influence readers thoughts, news reporting is extremely successful in telling readers what to think about. A process like Cultivation Analysis, composed of gradual influence made of repeated exposure, has the ability of directing readers attention to specific topics. An example of it can be found when considering how different is the presence of the "Green New Deal" (GND) mentions between Fox News and NPR. The topic is in fact found 35 times in headlines of the former against the 4 of the latter. It is crucial to specify that these Green New Deal-type programs were presented in 2018 by Representative Ocasio-Cortez (negatively mentioned by Fox News, according to the  sentiment analysis). As written by Lieven, the Green New Deal is "an essential step in winning over the working classes to support action on climate change", which requires a sense of unity that overcomes political views differences. Instead, Fox News presented the Green New Deal as a progressive initiative that would challenge traditional capitalist principles.

In November and December 2018, a few months after the proposal of the GND, a survey made by Yale and the Center for Climate Change Information at George Mason University showed that the support for it was strong among Democrats (92%), but also a large majority of Republicans (64%), including also conservative Republicans (57%),  were in support of the policy goals. This initial survey also showed how 82% of the respondents  had heard “nothing at all” about the Deal. The analysis made on the monthly trend of GND mentions in Figure 6 seems to prove this unawareness, since Fox News started talking about it in January 2019. 


<figure>
    <img src="../../assets/img/project_env/gnd.png" style="width: 80%;">
    <figcaption>Figure 6: Monthly trend of "Green New Deal" Mentions in Fox News Headline</figcaption>
</figure>



In the first months of 2019, the Green New Deal mentions in Fox News started to become more frequent, and a second survey, made in April 2019, showed how the public familiarity with it shifted dramatically. The proportion who had heard “nothing at all” about it decreased to 41%. Support for it remained high among Democrats, while support among Republicans dropped dramatically (-20 percentage points), especially among conservative Republicans (from 57% to 32%).

The April 2019 survey found that "82% of Republicans who watch, read, or listen to Fox News more than once per week have heard about the Green New Deal – a 64-point increase since December. Republicans who watch Fox News less often are substantially less familiar with the Green New Deal". In a similar way, support for the GND was lower among Republicans who watched Fox News more frequently than it was among Republicans who watched it less often: 22% against 56%. It is worth noting that frequent Fox News viewers constitute only 35% of Republican voters, and 56% of those Republicans who watched Fox News less frequently still supported the Green New Deal.

Concluding, the narrative frame used by Fox News had the ability to highly influence the Republican support of GND through a repeated exposure, leading to a huge reduction. This seems to confirm the idea behind the effects of framing on the human need to have an opinion on a specific topic, and the influence and contribution that repeated exposures have. As expressed by Mann, Fox News seems to have weaponized the GND accusing Representative Ocasio-Cortez of "Soviet-Style Propaganda", eroding the support for the proposal in just a few months.




#### 3.3: Applying Topic Modeling

<a id='topicappl'></a>

Analyzing the clusters obtained on the Fox News data, it is noticeable the recurrent appearance of political figures and public personalities. This factor seems to confirm how political discourse shapes the framing of environmental news, and how the reporting of these is made in order to ensure a favorable political climate, in line with the viewpoints of the media corporation.  This emphasis on individual actors is an attempt to reduce a complex and various world, trying to reconstruct it on a simpler model that exploits the prominence of a public person. The clusters not mentioning political figures (6, 8 and 10 in Figure 7) seem focused on some of the eight criteria that determine newsworthiness according to Yopp and McAdams, such as timeliness, proximity and emotional impact. The content is in fact wildfires and hurricanes in North America, and the attempt to ban gas stoves in California. 

Analyzing the topic modeling results from NPR's environmental coverage reveals instead a focus on  environmental concerns. The clusters appear to be less centered around individuals, with only one out of ten including a specific person (former US President Trump).
It's clear from these topics that NPR frames environmental issues from a more ecological and systemic perspective, with focus areas like renewable energy, waste management, biodiversity, and climate change.


<figure>
    <img src="../../assets/img/project_env/Screenshot (420).png" alt="Description of image" style="width: 80%;">
    <figcaption>Figure 7: Fox News Clusters generated by Topic Modeling</figcaption>
</figure>

<figure>
    <img src="../../assets/img/project_env/Screenshot (421).png" alt="Description of image" style="width: 80%;">
    <figcaption>Figure 8: NPR Clusters generated by Topic Modeling</figcaption>
</figure>


## Section 4: Conclusions

<a id='concl'></a>


It is evident that both media outlets use distinct strategies in framing environmental news. Fox News tends to be more politicized in its environmental coverage, frequently featuring political figures, their stances, and using a terminology that appears to deliberately set up an antagonistic stance towards people and topics. Conversely, NPR's focus is more ecological and systemic, with environmental topics taking center stage over individuals.

The sentiment analysis shows that the representation of the same topic can differ substantially across outlets. This divergence may potentially influence the perception and opinion formation of their respective audiences. The case of the "Green New Deal" is an example. Fox News' narrative framing was found to significantly impact Republican support for the policy proposal, emphasizing the power of media frames in shaping public opinion.

Topic modeling further revealed that Fox News often focuses on timely events, potentially aiming for immediate emotional impact. On the other hand, NPR emphasizes long-term environmental concerns.

By employing different narratives and focusing on distinct aspects of environmental news, media outlets can substantially influence public sentiment and understanding. This awareness calls for future research into the impact of media framing, as well as the development of strategies to communicate environmental news more effectively and impartially.

