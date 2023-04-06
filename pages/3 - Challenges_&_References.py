import streamlit as st


st.set_page_config(layout="wide")

st.title(":musical_note: Song Popularity Analysis :musical_note:")

st.markdown("**By: ARIAS Bright   |   QUINONES Domenica   |   RAMIREZ SANCHEZ Luisa   |   VERAYO Jea**")

st.markdown("---")

st.header(":mountain: Challenges")

st.markdown("""
            **1. Working in separate notebooks**  
            Working in Jupyter notebooks as a group requires manual consolidation of codes instead of working in a single notebook in real-time.
            
            **2. Incompatible packages with Streamlit**  
            Some NLP packages such as PyLDAvis do not work as seamlessly as in the Jupyter notebooks. Since the Streamlit notebook was created in Spyder, the dataframes were instead manually coded and charted to show the top words per topic.  
            
            **3. Random outputs from the LDA model**  
            As the group was optimizing the topics from the LDA modelling, they observed that the model outputs random topics each time the notebook was run.
            
            **4. Small dataset**  
            With only a little over 500 unique songs in the dataset, the analysis will have limitations in the words and topics from the lyrics.
            
            **5. Low resources for NER model**  
            Running the best model for NER take long time for giving the results of each song, therefore we used the “en_core_web_sm” model in the Streamlit app (Song analysis) for practical purposes. However, the best model is used for modeling the summary of all the songs.
            """, unsafe_allow_html=True)

st.markdown("---")

st.header(":books: References")

st.markdown("""
            Geeks for Geeks. (2021, October 07). Python | Sentiment Analysis using VADER. Retrieved from Geeks for Geeks: https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/
            
            Sievert, C., & Shirley, K. E. (2014, June 27). LDAvis: A method for visualizing and interpreting topics. Proceedings of the Workshop on Interactive Language Learning, Visualization, and Interfaces (pp. 63-70). Baltimore: Association for Computational Linguistics. Retrieved from https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf
            
            Uncover Hidden Insights: Advanced Named Entity Recognition. (2022, July 1). Retrieved from Express Analytics: https://www.expressanalytics.com/blog/what-is-named-entity-recognition-ner-benefits-use-cases-algorithms/
            
            """, unsafe_allow_html=True)