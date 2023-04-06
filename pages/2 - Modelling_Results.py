# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:58:10 2023

@author: mquinones
"""

############################# NLP Group Assignment ############################

#==============================================================================
# Import libraries
#==============================================================================
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import seaborn as sb
import matplotlib.colors as mcolors
#from gensim.corpora import Dictionary
#==============================================================================
# Tab 3: Modelling Results
#==============================================================================

# Upload dataset initially

results = pd.read_csv('C:\\Users\\jverayo\\Downloads\\NLP\\data\\results.csv')

# Layout -----

st.set_page_config(layout="wide")

st.title(":musical_note: Song Popularity Analysis :musical_note:")

st.markdown("**By: ARIAS Bright   |   QUINONES Domenica   |   RAMIREZ SANCHEZ Luisa   |   VERAYO Jea**")

st.markdown("---")

# Title -----

st.header(":chart_with_upwards_trend: Modelling Results")

st.markdown("<span style='font-size:16px;'>To identify if the different NLP analysis performed previously had any relation with the popularity of the songs, classification algorithms were trained and tested. Binary classification algorithms are used to predict the outcome of a binary response variable based on a set of predictor variables, in this case a popularity score higher than or equal to 60 is classified as 1, otherwise 0. In this report, the performance of various binary classification algorithms for predicting the song popularity in the dataset was explored. Specifically, the aim was to compare the effectiveness of different models with different variables including Decision Tree Classifier, Logistic Regression, Random Forest Classifier, Gradient Boosting Classifier, Support Vector Machine, Multi-Layer Perceptron Classifier, and K-Nearest Neighbors Classifier in terms of accuracy and AUC. To accomplish this goal, a series of experiments involving model training and testing evaluation were conducted. The results of this study will not only help to identify the best-performing algorithm but also provide insights into the strengths and weaknesses of each approach, which can inform future research in binary classification for NLP analysis.</span>", 
            unsafe_allow_html=True)

# Bar chart

# Generate a list of colors in the coolwarm palette
colors = sb.color_palette("coolwarm_r", 2)

# Convert RGB tuples to hex strings
hex_colors = [mcolors.rgb2hex(color) for color in colors]

# fig = px.bar(results, x="Models", y="Accuracy",
#              color='Model', barmode='group',
#              height=400, color_discrete_sequence=hex_colors)

# fig.update_yaxes(range=[0.5, 1])

# fig.data[0].text = [f"{acc:.4f}" for acc in results['Accuracy']]
# fig.data[0].textposition = 'outside'

fig = go.Figure()

for i, model in enumerate(results['Model'].unique()):
    # Filter the data for the current model
    data = results[results['Model'] == model]
    
    # Add a new trace for the current model
    fig.add_trace(go.Bar(
        x=data['Models'],
        y=data['Accuracy'],
        name=model,
        marker=dict(color=hex_colors[i]),
        text=[f"{acc:.4f}" for acc in data['Accuracy']],
        textposition='outside',
    ))

fig.update_yaxes(range=[0.5, 1])
fig.update_layout(height=400)

col1, col2, col3 = st.columns([1,6,1])

with col1:
    st.write("")

with col2:
    st.plotly_chart(fig, use_container_width=True)

with col3:
    st.write("")

st.subheader(":bulb: Conclusions")

st.markdown("<span style='font-size:16px;'>The first model, which includes all the original variables selected through Pearson Correlation and all text variables obtained throughout the analysis, performs better in terms of accuracy for most of the algorithms tested. The second model, which considers only the original variables selected through Pearson Correlation, obtains the highest accuracy rate with Gradient Boosted Tree, but all the other algorithms have lower accuracy than the ones in the first model. Therefore, it seems that including the text variables in the model leads to better accuracy for most of the algorithms and it may be beneficial to include the text variables in the model.</span>", 
            unsafe_allow_html=True)

st.markdown("<span style='font-size:16px;'>Nevertheless, it is important to be aware that accuracy is just one factor to consider when deciding whether to include the text variables in the model. Additional things such as interpretability and relevance of the text variables in the context of the lyrics should also be considered. Moreover, it may be useful to perform further analysis or experiments to confirm the results and assess the robustness of the models.</span>", 
            unsafe_allow_html=True)