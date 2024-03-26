#!/usr/bin/env python
# coding: utf-8

# In[1]:

#conetcting to pltoly for intractve grhaps
import plotly.offline as pyo
pyo.init_notebook_mode(connected=True)

# In[5]:

#importing important libraries to use for visualization
import pandas as pd
import plotly.graph_objects as go
from ipywidgets import widgets, interactive, Output
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import dash
from dash import Dash, dcc, html, Input, Output
from IPython.display import clear_output
from dash import html, dcc
import dash_core_components as dcc
import dash_html_components as html


# In[6]:


#importing Data from jupyter directory 
#if you will run the code you will need to take the data and uplaod it the jupiter
data = pd.read_csv("NamesE.csv")

#creating race bar chart for women names
#only run on ur computer if you have ffmpeg if not run it on google collab and it will work witg mp4
#the video is on you tube on my channel

#top 10 names in each year boys/girls
# copying the data
data4 = data.copy()

# Replacing '.' and '..' with 0 for numerical analysis
for column in data4.columns[1:]:
    try:
        data4[column] = data4[column].apply(lambda x: int(str(x).replace(',', '').replace('.', '0')))
    except ValueError:
        pass  # Handle non-numeric values here, for example setting them to NaN

app = Dash(__name__)
app = app.server
# Available years in the dataset for selection
years = [str(year) for year in range(1948, 2022)]  # Adjust based on your dataset

#drop down box for years
app.layout = html.Div([
    dcc.Dropdown(
        id='year-selector',
        options=[{'label': year, 'value': year} for year in years],
        value='2021'  # Default value
    ),
    dcc.Graph(id='top-boys-names'),
    dcc.Graph(id='top-girls-names')
])
#callbacks
@app.callback(
    [Output('top-boys-names', 'figure'),
     Output('top-girls-names', 'figure')],
    [Input('year-selector', 'value')]
)
def update_graph(selected_year):
    #dataset is prepared with correct gender labels
    boys_data = data4[data4['Gender'] == 'Male']
    girls_data = data4[data4['Gender'] == 'Female']

    # Converting counts to numeric, handling missing or special characters
    boys_data[selected_year] = pd.to_numeric(boys_data[selected_year], errors='coerce').fillna(0)
    girls_data[selected_year] = pd.to_numeric(girls_data[selected_year], errors='coerce').fillna(0)

    # top 10 names for boys and girls for the selected year
    top_boys = boys_data.nlargest(10, selected_year)[['Firstname', selected_year]]
    top_girls = girls_data.nlargest(10, selected_year)[['Firstname', selected_year]]

    # Createing bar plots for top names
    fig_boys = px.bar(top_boys, x='Firstname', y=selected_year, title=f"Top 10 Boys' Names in {selected_year}",
                      color_discrete_sequence=['#A3ADF8']*len(top_boys), pattern_shape_sequence=[""],  text_auto=True)
    #text size and color
    fig_boys.update_traces(textfont=dict(color='black', size=15))
    #corner style of the plots
    fig_boys.update_traces(go.Bar(marker=dict(cornerradius="30%")))
    
    fig_girls = px.bar(top_girls, x='Firstname', y=selected_year, title=f"Top 10 Girls' Names in {selected_year}",
                       color_discrete_sequence=['#EAA199']*len(top_girls), pattern_shape_sequence=[""],  text_auto=True)
     #text size and color
    fig_girls.update_traces(textfont=dict(color='black', size=15))
    #corner style of the plots
    fig_girls.update_traces(go.Bar(marker=dict(cornerradius="30%")))
    
    
    return fig_boys, fig_girls


#run the app
if __name__ == '__main__':
    app.run_server(debug=True, port= 7501)



