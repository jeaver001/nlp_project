#Import the lybraries
import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
from langdetect import detect
from nltk.corpus import stopwords
import spacy
import contractions
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.express as px
import matplotlib.colors as mcolors
from PIL import Image
#==============================================================================
# Tab 2: SOng Analysis
#==============================================================================

# Layout -----
st.set_page_config(layout="wide")


# Generate a list of colors in the coolwarm palette
colors = sb.color_palette("coolwarm_r", 3)

st.title(":musical_note: Song Analysis :musical_note:")

st.markdown("**By: ARIAS Bright   |   QUINONES Domenica   |   RAMIREZ SANCHEZ Luisa   |   VERAYO Jea**")

st.markdown("---")


#==============================================================================
# Song popularity
#==============================================================================

# Load the dataset
data = pd.read_csv(r'C:\Users\jverayo\Downloads\NLP\data\dataset.csv', 
                   encoding="Windows-1252", encoding_errors= 'replace')

#Replace columns of the topics name
data.rename(columns={"Topic1": "Heartbreak", "Topic2": "Going Out", "Topic3": "Street", "Topic4": "Regret & Desperation"}, inplace=True)

# Get a list of unique song names and artist names
song_names = sorted(data['song_name'].unique())

#Separate columns
Column1, Column2 = st.columns(2)

# Create a search text input for the song name
selected_song = Column1.selectbox('Search and select a song:', song_names, index=208)

# Get the list of unique artist names for the selected song
selected_song_artists = sorted(data[data['song_name'] == selected_song]['artist_name'].unique())

# Create a search text input for the artist name
selected_artist = Column2.selectbox('Search and select an artist:', selected_song_artists)
   
#Filter the dataset based on the selected song and artist
if selected_song and selected_artist:
    filtered_data = data[(data['song_name'] == selected_song) & (data['artist_name'] == selected_artist)]
    
    #==============================================================================
    # Information: Get the metrics of the songs
    #==============================================================================
    
    # Get the popularity of the selected song and artist
    
    #Rate of popularity of the music
    popularity = filtered_data['popularity'].values[0]
    
    #Number of artist_followers
    filtered_data['artist_followers'] = filtered_data['artist_followers'].map('{:,d}'.format)
    artist_followers = filtered_data['artist_followers'].values[0]
    
    #Max topic song
    topics = ["Heartbreak", "Going Out", "Street", "Regret & Desperation"]   
    
    #Generate the max value of topic per song
    topic_song = filtered_data[topics]
    def returncolname(row, colnames):
        return colnames[np.argmax(row.values)]
    
    topic_song['colmax'] = topic_song.apply(lambda x: returncolname(x, topic_song.columns), axis=1)
    name_topic = topic_song['colmax'].values[0]
    topic_song['max_topic'] = ((topic_song.max(axis=1))*100).round(2).astype(str) + '%'
    percentage_topic = topic_song['max_topic'].values[0]
    
           
# Row A for metrics
st.markdown('### Information ')

#Separate columns and display themù
col1, col2, col3, col4 = st.columns(4)
col1.metric("🟢 Popularity Score", popularity)
col2.metric("🟢 Number of subscribers", artist_followers)

df_sentiment = filtered_data["Sentiment"].values[0]
col3.metric("🟢 Sentiment", df_sentiment)

col4.metric("🟢 Topic Song", name_topic, percentage_topic + ' ' + 'max topic proportion')




#==============================================================================
# Preprocessing the data: For Word cloud and NER Analysis
#==============================================================================

# Convert the lyrics to string
lyrics_str = []

for index, row in filtered_data.iterrows():
    lyrics_str.append(str(row['lyrics']))
    
filtered_data['lyrics_str'] = lyrics_str


# Adding new columns
language = []
popularity_class = []

# Iterate over the rows of the DataFrame
for index, row in filtered_data.iterrows():
    language.append(detect(row['lyrics_str']))
    popularity_class.append(1 if row['popularity'] >= 60 else 0) 
    
# Assign the lists to new columns     
filtered_data['language'] = language
filtered_data['popularity_class'] = popularity_class

# Subset the df to english songs only
# Source: https://www.projectpro.io/recipes/add-custom-stopwords-and-then-remove-them-from-text
filtered_data = filtered_data[filtered_data['language'] == 'en'].reset_index()

#Import stopwords from nltk
nltk.download('stopwords')

lyrics_stopwords = ["intro", "bridge", "chorus", "verse", "outro", "whistling", "object", "length", "dtype", "name", "lyrics", 'ayy','huh','uh','ohoh',
                 'hey','le','la','ah','ha','haha','yo','ya','nah','ooh','mmm','eh','woah','oohooh','yeah','oh','lalalalalala', 'na', 'eheh']

stpwrd = nltk.corpus.stopwords.words('english')
stpwrdsongs = set(stopwords.words('english')).union(lyrics_stopwords)


#Create a preprocessing function to apply to the lyrics
# Source: Class notebook S2 Text_pre_processing_complete
wpt = nltk.WordPunctTokenizer() 
nlp = spacy.load('en_core_web_sm')
nlp.disable_pipes('ner')

def clean_lyrics_fun(song):
    # Remove text within brackets
    song = re.sub(r'\[.*?\]', '', song)
    # Lower case and remove special characters\whitespaces
    song = re.sub(r'\b[a-zA-Z]\b', '', song)
    song = re.sub(r'[^\w\s]', '', song)
    song = song.lower()
    # Remove extra spaces
    song = re.sub(' +', ' ', song)
    # Expand contractions
    song = contractions.fix(song)
    # Additionaly: remove trailing and leading blanks
    song = song.strip()
    # tokenize document
    tokens = wpt.tokenize(song)
    # Removing numeric
    tokens = [token for token in tokens if not token.isnumeric()]
    # Filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stpwrdsongs]
    # Lemmatization
    ## Create a song object
    song = nlp(' '.join(filtered_tokens))
    ## Generate lemmas
    text_lemmatized = [token.lemma_ for token in song]
    #text_lemmatized = [wordnet_lemmatizer.lemmatize(token) for token in doc]
    return text_lemmatized


# Apply the function to the lyrics
clean_lyrics = [clean_lyrics_fun(r) for r in filtered_data['lyrics_str']]

# Convert to list
lyrics_list = [' '.join(word) for word in clean_lyrics]
filtered_data['lyrics_list'] = lyrics_list


#==============================================================================
# NER Name identities analysis Lyrics
#==============================================================================

# display the NER Analysis and the NER name identities
nerEntities, nerAnalysis  = st.columns(2)

with nerEntities:
    
    ###chat
    #Convert the lyrics in string    
    lyrics_string = '\n'.join([''.join(map(str, sublist)) for sublist in filtered_data['lyrics_str']])
    
    # Load a pre-trained model
    ner = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'matcher'])
    
    # Create a new document: doc and process the text with the model
    doc = ner(lyrics_string)
    
    # Generate HTML code using displacy.serve()
    html_code = spacy.displacy.render(doc, style='ent')

#Display the Lyrics, Word cloud and Sentiment Analysis side by side
lyrics, others = st.columns(2)

with lyrics:
    #Title
    st.markdown("<h3 style='text-align: center;'>📌 Lyrics Named Entities</h3>", unsafe_allow_html=True)
    # Display the HTML code
    st.write("In the following you can see the named entities recognition in the song:")
    
    st.markdown(html_code, unsafe_allow_html=True)
    
#==============================================================================
#
# Word Cloud
#
#==============================================================================
#plt.gcf() from https://discuss.streamlit.io/t/how-to-add-wordcloud-graph-in-streamlit/818/14

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

#the cloud
wordcloud(filtered_data)
#st.pyplot(plt.gcf())

#Display the Word cloud and Sentiment Analysis side by side
wordCloud, sentAnalysis = st.columns(2)

with others:
    #Title
    st.markdown("<h3 style='text-align: center;'> 📌 Word Cloud</h3>", unsafe_allow_html=True)
    #Explanation of the word cloud
    st.write("Below is displaying the importance of song's lyrics:")
    #Plot
    st.pyplot(plt.gcf())

#==============================================================================
# NER Analysis
#==============================================================================
   
#Select the columns
ner = ["SWEAR", "TIME", "SLANG", "PRODUCT", "GPE", "PERSON", "ORG"]

#Filter the data with those columns
ner_song = filtered_data[ner]

#Select the first values of row
row = ner_song.iloc[0]

# generate a list of colors in the coolwarm palette
colors = sb.color_palette("coolwarm_r", len(topics))

# convert RGB tuples to hex strings
hex_colors = [mcolors.rgb2hex(color) for color in colors]

# Create a new DataFrame with the values of the first row and the variable names as columns
df_ner = pd.DataFrame({'Variable': row.index, 'Value': row.values})

# Create a treemap of the values in the new DataFrame
fig = px.treemap(df_ner, path=['Variable'], values='Value', color_discrete_sequence=hex_colors)

#Display with the NER distribution
with others:
    st.markdown("<h3 style='text-align: center;'>📌 Named Entity Recognition Analysis</h3>", unsafe_allow_html=True)
    st.write('<p class="cursive">The image below shows the distribution of the named entities recognition in the song:</p>', unsafe_allow_html=True)
    st.plotly_chart(fig)

#==============================================================================
# References
#==============================================================================
#https://stackoverflow.com/questions/39874501/get-column-names-for-max-values-over-a-certain-row-in-a-pandas-dataframe
#https://stackoverflow.com/questions/43102734/format-a-number-with-commas-to-separate-thousands