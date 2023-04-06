import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.express as px
import matplotlib.colors as mcolors
    
st.set_page_config(layout="wide")

st.title(":musical_note: Song Popularity Analysis :musical_note:")

st.markdown("**By: ARIAS Bright   |   QUINONES Domenica   |   RAMIREZ SANCHEZ Luisa   |   VERAYO Jea**")

st.markdown("---")

st.header(":rocket: Goal of the Project")
st.markdown("""
            The goal of predicting a song's popularity based mostly on its lyrics involves using Natural Language Processing 
            techniques to analyze the lyrics of a song and determine the likelihood of its commercial success. 
            The success of a song can be measured in a variety of ways, in this case, a popularity score from 1 to 
            100 is provided, being 100 the most recognized song.
            
            To predict the song's success based mostly on its text, several NLP models have analyzed the lyrics for
            various features, such as the sentiment, topic, entities and style of the language used. Additionally, 
            other factors regarding the rhythm or the artist are considered.
            
            The success of a song is hard to predict, and many factors contribute to its popularity. However, 
            analyzing the lyrics of a song can provide valuable insights into its potential success and help music 
            industry professionals make more informed decisions about which songs to promote and invest in.
            """, unsafe_allow_html=True)

st.markdown("---")

st.header(":clipboard: About the Dataset")
st.markdown('The dataset contains 639 songs from Spotify, which includes information on the music, indexes of popularity, artist information and the lyrics (scraped from Genius).')

st.markdown("---")

st.header(":hammer: Preprocessing the Lyrics")
st.markdown("""
            In the original dataset, the song lyrics are displayed in the typical format of lines and stanzas. To conduct textual analysis and machine learning models, the following preprocessing steps were performed to the lyrics:
            1. **Removal of duplicate songs.** The dataset contained duplicate rows of songs which have multiple artists. As artist names will not be used in the modelling and analysis, songs with duplicate IDs were omitted.
            2.	**Removal of non-English songs.** Songs that are not in English were removed instead of translated as they only comprise a very little portion of the dataset.
            3.	**Converting the lyrics to string type.** The original column was in object type and must therefore be converted to string to perform the proceeding steps.
            4.	**Cleaning the texts**
                
                a.	Removal of line breaks and words in brackets such as Chorus and Instrumental
                
                b.	Removal of numbers, white spaces, and special characters
            5.	**Text normalization by changing the text to lowercase.**
            6.	**Simplification**
            
                a.	Expanding contractions in songs such as I’ll to I will
                
                b.	Removal of stopwords. The NLTK English stopwords were used and expanded to song-related stopwords 
                such as ‘ooh’ and ‘la’ (performed after tokenizing the text)
            7.	**Tokenization.** NLTK’s WordPunctTokenizer was used to split the lyrics into tokens.
            8.	**Lemmatization**
            """, unsafe_allow_html=True)

#==============================================================================
# Importing the data
#==============================================================================

data = pd.read_csv(r'C:\Users\jverayo\Downloads\NLP\data\dataset.csv', 
                   encoding="Windows-1252", encoding_errors= 'replace')

#==============================================================================
#
# Sentiment Analysis
#
#==============================================================================
st.markdown("---")

st.header(':chart_with_upwards_trend: Analysis')

st.markdown("---")


st.subheader(":revolving_hearts: Sentiment Analysis")

st.markdown("""
            Sentiment Analysis is the process of determining the “sentiment” of a text, usually classified as 
            positive, negative or neutral. For the lyrics analysis, the VADER sentiment analyzer was used as 
            it returns a positive, neutral and negative score for each song, as well as a compound score, which 
            determines the overall sentiment of the lyrics. This score obtains the normalized sum of all the 
            lexicon ratings, between -1 (negative) and +1 (positive).  
            
            Although VADER is a lexicon and rule-based sentiment analysis tool specifically designed to analyze 
            sentiment in social media texts, positive results were found using it for lyrics sentiment analysis 
            because both, social media texts and lyrics often contain figurative language, jargon, sarcasm, and 
            cultural references, making them difficult to analyze using traditional sentiment analysis methods. <sup>1</sup>
            
            In this tree map it is possible to see that most songs have a positive context as an 
            overall sentiment score (62%), followed by the negative songs with 31% of the songs. Not many songs 
            were determined as neutral because most of them are instrumental or their lyrics were short, which 	
            makes it hard to determine the compound sentiment.
            
            The bar chart shows the average popularity score of each sentiment (positive, neutral and negative). 
            There is not a big difference between each sentiment, although songs categorized as positive are more 
            popular than the ones rated as negative or neutral.  
            """, unsafe_allow_html=True)

# calculate sentiment count
sentiment_groups = data.groupby('Sentiment')['id'].agg(['count']).reset_index(drop=False)

# generate a list of colors in the coolwarm palette
colors = sb.color_palette("coolwarm_r", len(sentiment_groups))

# convert RGB tuples to hex strings
hex_colors = [mcolors.rgb2hex(color) for color in colors]

# create treemap
fig1 = px.treemap(sentiment_groups, path=['Sentiment'], values='count', 
                  color_discrete_sequence=hex_colors)

# create barplot
sent_pop_groups = data.groupby('Sentiment')['popularity'].agg(['mean']).reset_index()
sent_pop_groups['mean'] = round(sent_pop_groups['mean'], 0)

sb.set_style("darkgrid")
sb.set(font_scale=0.5)
fig2, ax = plt.subplots(figsize=(4, 2))
sb.barplot(data=sent_pop_groups, x="Sentiment", y="mean", palette="coolwarm", ax=ax)
ax.bar_label(ax.containers[0])
ax.set_ylim(1, 100)
ax.set(ylabel='Popularity')

# display the figures side by side
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h3 style='text-align: center;'>Sentiment Distribution</h3>", unsafe_allow_html=True)
    st.plotly_chart(fig1)
with col2:
    st.markdown("<h3 style='text-align: center;'>Mean Popularity by Sentiment</h3>", unsafe_allow_html=True)
    st.pyplot(fig2)

vader_source = 'https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/'
st.caption("1 [(Geek for Geeks, 2021)](%s)" % vader_source)
#==============================================================================
#
# Word Cloud
#
#==============================================================================
#plt.gcf() from https://discuss.streamlit.io/t/how-to-add-wordcloud-graph-in-streamlit/818/14
st.markdown("---")

st.subheader(":cloud: Word Cloud")

st.markdown("""
            A **word cloud** is a visualization tool used to display without prearrangement the most frequent words 
            in a text. Each word will have varying font sizes depending on its frequency in the text, with the most 
            frequent word being the largest word and the least frequent word being the smallest word in the cloud. 
            Word clouds were used to compare the most frequent words in the lyrics of popular songs vs. unpopular songs.
            """, unsafe_allow_html=True)

#Creating a function to generate wordclouds
def wordcloud(df):
    lyrics_string = df['lyrics_list'].str.cat(sep=', ')
    wc = WordCloud(width = 800, height = 400,
                      background_color ='white',
                      colormap = 'coolwarm',
                      min_font_size = 14)
    wc.generate(lyrics_string)
    # plot the WordCloud image                      
    plt.figure(figsize = (4, 4), facecolor = None)
    plt.imshow(wc)
    plt.axis("off")
    plt.tight_layout(pad = 0)

# Creating subsets 
popular_df = data[data['popularity_class']==1]
unpopular_df = data[data['popularity_class']==0]

#Instantiating the word clouds

st.markdown("#### **Word Cloud Per Sentiment**")

positive = data[data['Sentiment'] == 'Positive']
negative = data[data['Sentiment'] == 'Negative']

st.markdown("""
            As seen below, Positive songs contain the words 'know', 'love', 'want', 'say', ‘tell’ and 'go'. 
            These words imply songs about expressing one’s thoughts and feelings most likely about love and desire.  
            Results of the topic analysis will later be discussed to deep dive on the most relevant lyrical themes in the dataset.
            
            Meanwhile, curse words such as 'nigga', 'bitch', and 'fuck' seem to be more frequent in Negative songs. 
            """, unsafe_allow_html=True)

col_pos_wc, col_neg_wc, = st.columns(2)

# =============================================================================
with col_pos_wc:
    st.markdown("#### **Positive Songs**")
    wordcloud(positive)
    st.pyplot(plt.gcf())
    
with col_neg_wc:
    st.markdown("#### **Negative Songs**")
    wordcloud(negative)
    st.pyplot(plt.gcf())
# =============================================================================
   
#==============================================================================
#
# Topic Modelling
#
#==============================================================================
st.markdown("---")

st.subheader(":speech_balloon: Topic Analysis: Song Themes")

lda_source = "https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf"

st.markdown("""
            Songs are an artistic form of expression of one's experiences, thoughts, and feelings. What a song is about 
            or how much a person can relate to its message can sometimes be a reason for liking or disking it. It would 
            therefore be interesting to explore the most common themes that are sung about and investigate if themes do 
            contribute to the popularity of a song.
            
            Latent Dirichlet Allocation or LDA was used to dissect the words in the lyrics of each song to identify the 
            most usual topics in the dataset as well as the most evident topic for each song. In the process of optimization, 
            words that appeared in less than 5% as well as in more than 50% of the lyrics were removed. Ultimately, the model 
            generated four main topics that did not have any overlaps with the other topics. 
            
            The table below shows four identified topics, including the top 10 words that are most relevant to 
            the topic. In generating the words, lambda was set to 0.1 in order to show the terms with a higher lift for the 
            topic.<sup>2</sup> Based on these top words, the following topics were identified.
            """, unsafe_allow_html=True)

top_words_per_topic = pd.read_csv('C:\\Users\\jverayo\\Downloads\\NLP\\data\\top_words_per_topic.csv')

col1_topic, col2_topic, = st.columns([0.3, 0.7])


with col1_topic:
    st.markdown("<h3 style='text-align: center;'>Top 10 Words Per Topic</h3>", unsafe_allow_html=True)    
    st.table(top_words_per_topic)

with col2_topic:
    st.markdown("""
                **Topic 1: Heartbreak** :broken_heart:
                
                Two concepts can be derived from this topic: some of the words relate to (1) hurting and losing someone 
                including "without", "cry", "die", “late” and "tear", while some of the words relate to (2) wishful thinking 
                including "remember", "dream", "life", and "happy". Combining these concepts suggests of thoughts, feelings, 
                and actions of someone heartbroken.  
                Song with Highest Relevance: Loved By You (by Justin Bieber)
                
                **Topic 2: Going Out** :crescent_moon:
                
                In this topic, there are more words referring to activities including "dance", "come", "move", and "drive". It also includes words about spending the night out like "tonight", "long", "place". Party beats will most likely fall into this topic.  
                Song with Highest Relevance: Levitating (by Dua Lipa)
                
                **Topic 3: Street** :chains:
                
                This topic includes more swear and slang words which most likely includes rap and hip-hop songs.  
                Song with Highest Relevance: Can't Stop (by Da Baby)
                
                **Topic 4: Regret & Desperation** :pensive:
                
                This topic is a combination of uncertainty (“maybe”, “wish”), reaching out (“call”, “talk”, “wait”), and strong urges (“fire”, “kill”). This can be songs about feelings or actions after losing someone or something.   
                Song with Highest Relevance: If I Can't Have You (by Shawn Mendes)
                """, unsafe_allow_html=True)

st.caption("2 [(Sievert and Shirley, 2014)](%s)" % lda_source)



data = data.rename(columns={'Topic1': 'Heartbreak', 'Topic2': 'Going Out', 'Topic3': 'Street', 'Topic4': 'Regret & Desperation'})
data['max_topic_perc'] = data[['Heartbreak', 'Going Out', 'Street', 'Regret & Desperation']].max(axis=1)
data['topic'] = data.apply(lambda row: 'Heartbreak' if row['max_topic_perc'] == row['Heartbreak'] else 'Going Out' if row['max_topic_perc'] == row['Going Out'] else 'Street' if row['max_topic_perc'] == row['Street'] else 'Regret & Desperation', axis=1)
topic_count = pd.DataFrame(data.groupby(['popularity_class','topic'])['id'].count())

topic_count = topic_count.reset_index()
fig3, ax = plt.subplots(figsize=(4, 2))
sb.barplot(data=topic_count, x="topic", y="id", hue='popularity_class', palette="coolwarm", ax=ax)
ax.set_ylim(1, 150)
ax.set(ylabel='Song Count')
ax.set(xlabel='Topics')
ax.bar_label(ax.containers[0])
#https://www.geeksforgeeks.org/how-to-show-values-on-seaborn-barplot/
for i in ax.containers:
    ax.bar_label(i,)


col3_topic, col4_topic, = st.columns([0.4, 0.4])

with col3_topic:
    st.markdown('### **Topic Distribution by Popularity Class**')
    st.markdown("""
                Overall, there are more popular songs than unpopular songs in the dataset based on the popularity score. However, when the number of all songs in each class are compared by popularity class, it can be seen in the chart on the right that there are more unpopular songs in the Street Topic. 
                """, unsafe_allow_html=True)
with col4_topic:
    st.pyplot(fig3)
#==============================================================================
#
# NER
#
#==============================================================================
st.markdown("---")

st.subheader(":mag: NER")

st.markdown("""
            **Name Entity Recognition (NER)** is a subfield of NLP that involves identifying and 
            categorizing named entities in text. This process involves two steps: detecting 
            entities in the text and classifying them into categories. NER is important because 
            it helps extract useful information from unstructured data and is essential for 
            dealing with large datasets. Some of the categories that NER focuses on include person, 
            organization, and place/location, as well as date/time, numeral measurements, and email addresses. <sup>3</sup>
            
            •	**Annotate data:** Entities of 150 randomly selected songs were identified with the tool LightTag, which is 
            a free text annotation tool for teams. This step was done in order to create a costume model that identifies 
            the type of vocabulary present in the song. The chosen entities to analyze the lyrics are: person (PERSON), 
            organization (ORG), time, product, swear words and slang. 
            
            •	**Prepare the data:** Once the dataset was annotated, a JSON file was generated. This file contains the lyrics 
            and its entities, needed to train the model. 
            
            •	**Choose a machine learning framework:** Next, the spaCy machine learning framework was selected to train the model.
            
            •	**Define the model architecture:** The model architecture was defined using the recommended configuration of spaCy 
            for training NER models, which at the same time is based on the machine learning library Thinc.
            
            •	**Train the model:** After the model architecture was defined, the training of the model started on the annotated 
            dataset (JSON file) which contains 150 songs. The model was trained using 500 batches and 50 epochs to feed the 
            training data through the model, adjusting the model's weights and biases to minimize the error between the predicted 
            and actual tokens (toks2vec) and entity labels (NER).
            
            •	**Evaluate and tune the model:** To evaluate the performance of the model during training, spaCy generates the loss
            for both parts of the pipeline (tok2vec and NER). Additionally, it provides evaluation metrics such as F-score, precision,
            recall, and accuracy to gain insights into the model's performance. These metrics allow to adjust and improve the model's
            performance, ensuring that it accurately learns the patterns in the data. In this sense, to improve the performance and
            the efficiency of the model a bigger training set was provided, as well as a decrease in the number of epochs (from 300 to 50)
            and the number of batches (from 1000 to 500). Thus, the best model obtained a score of 0.98.
            
            •	**Deploy and use the model:** Once the best model was trained it was applied to the dataset which contains the clean lyrics.

            """, unsafe_allow_html=True)

ner_source = 'https://www.expressanalytics.com/blog/what-is-named-entity-recognition-ner-benefits-use-cases-algorithms/#:~:text=Named%20Entity%20Recognition%20(NER)%20is,%2C%20locations%2C%20and%20so%20on.'
st.caption("3 [(Express Analytics, 2022)](%s)" % ner_source) 

long_format=pd.melt(data, id_vars='id', value_vars=['SWEAR', 'TIME', 'SLANG', 'PRODUCT','GPE','PERSON', 'ORG'])
entities_grouped = long_format.groupby('variable').sum().reset_index(drop=False)
entities_grouped=entities_grouped.rename(columns={0: "entities", 1: "sum"}).reset_index(drop=True)
entities_grouped= entities_grouped.sort_values("value", ascending=False)

# Create a barplot using seaborn
sb.set(style="white")
sb.set(font_scale=0.5)
ent = sb.barplot(x='value', y="variable", data=entities_grouped, palette='coolwarm')
ent.set(xticklabels=[])
ent.legend_.remove()
#ent.set(legend=False)
# Add chart title and axis labels
plt.xlabel("Count")
plt.ylabel("Entities")

total = entities_grouped['value'].sum()
for p in ent.patches:
    ent.annotate(f'{p.get_width():.0f} ({p.get_width()/total*100:.1f}%)',
                 (p.get_x() + p.get_width(), p.get_y() + p.get_height() / 2),
                 ha='left', va='center')


col1_ner, col2_ner, = st.columns([0.6, 0.4])

with col1_ner:
    st.markdown("<h3 style='text-align: center;'>Number of Entities</h3>", unsafe_allow_html=True)
    st.pyplot(ent.figure)

with col2_ner:
    st.markdown("""
                Utilizing entity recognition enables us to identify crucial information in songs. Through this method, we can discern that terms pertaining to time and people are among the most significant entities in the song. On the contrary, entities such as organizations (ORG) and places (GPE) are mentioned less frequently 
                """, unsafe_allow_html=True)