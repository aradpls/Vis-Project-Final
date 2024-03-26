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


# In[13]:


#Percentage Timeline Trend for Multiple Names
# Manipulting the data for comfort not using .. and . turning them to zero this amount is not accurate and can be five to ten
df_plotly = data.copy()
df_plotly.replace(['.', '..'], np.nan, inplace=True)
df_plotly.replace({',': ''}, regex=True, inplace=True)
df_plotly.iloc[:, 3:-2] = df_plotly.iloc[:, 3:-2].apply(pd.to_numeric, errors='coerce')
total_counts_per_year = df_plotly.iloc[:, 3:-2].sum(axis=0)

# Activeating the Dash app
app1 = dash.Dash(__name__)
#for showing in html
app1.layout = html.Div([
    # Dropdown menu for selecting gender
    dcc.Dropdown(
        id='gender-dropdown',
        options=[
            {'label': 'Male', 'value': 'Male'},
            {'label': 'Female', 'value': 'Female'}
        ],
        value='Male'  # Default value
    ),
    
    # Plotly graph output
    dcc.Graph(id='name-graph')
])
# Define the callback to update the graph
@app1.callback(
    Output('name-graph', 'figure'),
    [Input('gender-dropdown', 'value')]
)
def update_graph(selected_gender):
    fig1 = go.Figure()
    #top ten names of all times for both of the genders in array
    names_to_plot = {
        'Male': ["אורי", 'דניאל', 'אחמד', "יצחק", "יעקב", "אברהם", "יוסף", "משה", "דוד", "מוחמד"],
        'Female': ["מאיה", 'חנה', 'שירה', "תמר", "אסתר", "שרה", "רחל", "יעל", "מיכל", "נועה"]
    }[selected_gender]
    #Decideing on how to show the graphs and caculation behind it
    for name in names_to_plot:
        filtered_df = df_plotly[df_plotly['Firstname'].str.strip().str.lower() == name.lower()]
        if not filtered_df.empty:
            name_data = filtered_df.iloc[:, 3:-2].sum(axis=0)
            name_percentage = (name_data / total_counts_per_year) * 100 #prentge from all names in a specific year
            years = df_plotly.columns[3:-2].astype(str)
            fig1.add_trace(go.Scatter(x=years, y=name_percentage, mode='lines+markers', name=name))
    
    fig1.update_layout(
        title=f'Top Ten Most Popular Names Of All Time ({selected_gender})',
        xaxis_title='Year',
        yaxis=dict(title='Percentage of Total Names'),
        xaxis=dict(tickmode='linear', dtick=10),
        legend_title='First Name'
    )
    
    return fig1 #Return the grpah we buld

if __name__ == '__main__':
    app1.run_server(debug=True, port=7511) #Show it on the app


# In[ ]:


#Name Popularity Over Time by Gender
# Manipulting the data for comfort not using .. and . turning them to zero this amount is not accurate and can be five to ten
dataarea = data.copy()
dataarea.replace(['.', '..'], 0, inplace=True)
dataarea.replace({',': ''}, regex=True, inplace=True)
for col in dataarea.columns[3:-2]: #3rd to the last-2 columns are the years
    dataarea[col] = pd.to_numeric(dataarea[col], errors='coerce', downcast='integer')

data_melted = dataarea.melt(id_vars=['Firstname', 'Gender', 'Religion'],
                        value_vars=[str(year) for year in range(1948, 2022)],
                        var_name='Year', value_name='Count')

# Converting 'Year' to an integer right after melting
data_melted['Year'] = data_melted['Year'].astype(int)

# Converting 'Count' to numeric types and replace NaNs with zeros
data_melted['Count'] = pd.to_numeric(data_melted['Count'], errors='coerce').fillna(0)

# Grouping the data by 'Firstname', 'Gender', and 'Year' and sum across all religions for no duplicates
data_grouped = data_melted.groupby(['Firstname', 'Gender', 'Year'], as_index=False).sum()

# Filtering names that appear in both genders
names_in_both_genders = data_grouped.groupby('Firstname').filter(lambda x: len(x['Gender'].unique()) > 1)['Firstname'].unique()

# Initialize the Dash app with a title
app2 = dash.Dash(__name__, title="Name Popularity Over Time")

# Define consistent colors for genders
gender_colors = {
    'Male': '#A3ADF8',  
    'Female': '#EAA199',  
}

# Generateing marks for the slider
def generate_slider_marks(start, end):
    marks = {year: {'label': str(year), 'style': {'text-align': 'left'}} for year in range(start, end + 1, 10)}
    marks[start] = str(start)
    marks[end] = str(end)
    return marks

# Define the app layout
#creating dropdown list with names from both genders
app2.layout = html.Div([
    html.Div([
        html.Label('Name:'),
        dcc.Dropdown(
            id='name-dropdown',
            options=[{'label': name, 'value': name} for name in names_in_both_genders],
            value=names_in_both_genders[0]
        ),
        html.Br(),#year slider with jumps of 10 
        html.Label('Year Range:'),
        dcc.RangeSlider(
            id='year-slider',
            min=1948,
            max=2021,
            value=[1948, 2021],
            marks=generate_slider_marks(1948, 2021),
            
        ),
        html.Br(), #for the iser to decide id he wants the graph stacked or not
        dcc.Checklist(
            id='stacked-checklist',
            options=[{'label': 'Stacked Plot?', 'value': 'stacked'}],
            value=['stacked']
        ),
    ]),
    dcc.Graph(id='trend-graph')
])

# Define the callback to update the graph
@app2.callback(
    Output('trend-graph', 'figure'),
    [Input('name-dropdown', 'value'),
     Input('year-slider', 'value'),
     Input('stacked-checklist', 'value')]
)
def update_graph(selected_name, selected_years, checklist_values): #updaing the grpah for each action the user does
    filtered_data = data_grouped[
        (data_grouped['Firstname'] == selected_name) &
        (data_grouped['Year'] >= selected_years[0]) &
        (data_grouped['Year'] <= selected_years[1])
    ]
    #what to show and the grhaps lines anf how to fill them for area chart
    fig = go.Figure()
    for gender in ['Male', 'Female']:
        df_filtered_by_gender = filtered_data[filtered_data['Gender'] == gender]
        if 'stacked' in checklist_values:
            fig.add_trace(go.Scatter(
                x=df_filtered_by_gender['Year'],
                y=df_filtered_by_gender['Count'],
                mode='lines',
                name=gender,
                stackgroup='one',
                line=dict(color=gender_colors[gender])
            ))
        else:
            fig.add_trace(go.Scatter(
                x=df_filtered_by_gender['Year'],
                y=df_filtered_by_gender['Count'],
                mode='lines',
                name=gender,
                fill='tozeroy',
                line=dict(color=gender_colors[gender])
            ))
    #what to show on the grph axis, legend , title , intervlal of yars
    fig.update_layout(
        title=f"Name Popularity Over Time by Gender: '{selected_name}'",
        xaxis_title='Year',
        yaxis_title='Count',
        xaxis=dict(
            tickmode='linear',
            tick0=1948,
            dtick=10
        ),
        legend_title='Gender'
    )

    return fig

# Run the app
if __name__ == '__main__':
    app2.run_server(debug=True, port=7520)




# In[ ]:


# copying the data 
data1 = data.copy()

# Replacing '.' and '..' with 0 for numerical analysis
for column in data1.columns[1:]:
    try:
        data1[column] = data1[column].apply(lambda x: int(str(x).replace(',', '').replace('.', '0')))
    except ValueError:
        pass  # Handle non-numeric values here, for example setting them to NaN
 
# Group the data by 'Firstname', 'Gender', and 'Religion', and calculate the sum of 'SumYears' for each group
grouped_data = data1.groupby(['Firstname', 'Gender', 'Religion'])['SumYears'].sum().reset_index()

# Pivoting the data to get 'Religion' as columns and 'Firstname' and 'Gender' as rows, with 'SumYears' as values
pivot_data = grouped_data.pivot_table(index=['Firstname', 'Gender'], columns='Religion', values='SumYears', fill_value=0)

# Reseting the index to make 'Firstname' and 'Gender' columns again
pivot_data.reset_index(inplace=True)

# Calculateing the number of unique religions for each name
pivot_data['NumReligions'] = pivot_data.iloc[:, 2:].gt(0).sum(axis=1)

# Filtering the DataFrame to include only names that appear in all religions
filtered_data = pivot_data[pivot_data['NumReligions'] >= 4]

# Reseting the index to make 'Firstname' and 'Gender' columns again
filtered_data.reset_index(drop=True, inplace=True)

# Droping the 'NumReligions' column as it's no longer needed
filtered_data.drop(columns='NumReligions', inplace=True)

# Get the number of names left after filtering
num_names_left = len(filtered_data)

#using our pivot_data
correlation_matrix = pivot_data[['Christian', 'Druze', 'Jew', 'Muslim']].corr()

# Plot the correlation matrix as a heatmap with custom color scale
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', center=0, fmt=".2f")
plt.title('Correlation Matrix For Common Names Between Religions')
plt.show()


# In[ ]:


# copying the data
data2 = data.copy()

# Replacing '.' and '..' with 0 for numerical analysis
for column in data2.columns[1:]:
    try:
        data2[column] = data2[column].apply(lambda x: int(str(x).replace(',', '').replace('.', '0')))
    except ValueError:
        pass  # Handle non-numeric values here, for example setting them to NaN

# Group the data by 'Firstname', 'Gender', and 'Religion', and calculate the sum of 'SumYears' for each group
grouped_data_2 = data2.groupby(['Firstname', 'Gender', 'Religion'])['SumYears'].sum().reset_index()

# Pivot the data to get 'Religion' as columns and 'Firstname' and 'Gender' as rows, with 'SumYears' as values
pivot_data_2 = grouped_data_2.pivot_table(index=['Firstname', 'Gender'], columns='Religion', values='SumYears', fill_value=0)

# Reset the index to make 'Firstname' and 'Gender' columns again
pivot_data_2.reset_index(inplace=True)

# Calculate the number of unique religions for each name
pivot_data_2['NumReligions'] = pivot_data_2.iloc[:, 2:].gt(0).sum(axis=1)

# Filter the DataFrame to include only names that appear in all religions
filtered_data_2 = pivot_data_2[pivot_data_2['NumReligions'] >= 4]

# Reset the index to make 'Firstname' and 'Gender' columns again
filtered_data_2.reset_index(drop=True, inplace=True)

# Drop the 'NumReligions' column as it's no longer needed
filtered_data_2.drop(columns='NumReligions', inplace=True)

# Get the number of names left after filtering
num_names_left_2 = len(filtered_data_2)

# Calculate the minimum value from the religion columns
filtered_data_2['MinReligion'] = filtered_data_2[['Christian', 'Druze', 'Jew', 'Muslim']].min(axis=1)

# Calculate the total number of religions
filtered_data_2['TotalReligions'] = filtered_data_2[['Christian', 'Druze', 'Jew', 'Muslim']].sum(axis=1)


# Calculate the maximum value from the religion columns
filtered_data_2['MaxReligion'] = filtered_data_2[['Christian', 'Druze', 'Jew', 'Muslim']].max(axis=1)

# Perform the division to get the desired calculation
filtered_data_2['Calculation'] = filtered_data_2['MinReligion'] / filtered_data_2['MaxReligion']
filtered_data_2 = filtered_data_2.sort_values(by='Calculation', ascending=False)

# Assuming 'filtered_data_2' is your DataFrame
# Add a new column 'num_of_letters' to 'filtered_data_2'
filtered_data_2['num_of_letters'] = filtered_data_2['Firstname'].apply(lambda x: len(x))

print("Number of names left:", num_names_left_2)

# Create a new DataFrame 'letters_data' containing selected columns from 'filtered_data_2'
letters_data = filtered_data_2[['Firstname', 'Christian', 'Druze', 'Jew', 'Muslim', 'num_of_letters']].copy()



# In[ ]:


# to show top 5 names that are in in relgions and common
top_5 = filtered_data_2.head(5)

# Define categories (religions)
categories = ['Religion', 'Christian', 'Druze', 'Jew', 'Muslim']

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.Label("Select Firstname:"),
    dcc.Dropdown(
        id='firstname-dropdown',
        options=[{'label': firstname, 'value': firstname} for firstname in top_5['Firstname'].unique()],
        value=[top_5['Firstname'].iloc[0]],  # Set default value as a list
        multi=True
    ),
    html.Label("Select Gender:"),
    dcc.Dropdown(
        id='gender-dropdown',
        options=[{'label': gender, 'value': gender} for gender in top_5['Gender'].unique()],
        value=[top_5['Gender'].iloc[0]],  # Set default value as a list
        multi=True
    ),
    dcc.Graph(id='spider-chart')
])

# Define callback to update the spider chart based on filter selections
@app.callback(
    Output('spider-chart', 'figure'),
    [Input('firstname-dropdown', 'value'),
     Input('gender-dropdown', 'value')]
)
def update_spider_chart(selected_firstnames, selected_genders):
    # Ensure that selected_firstnames and selected_genders are lists
    if not isinstance(selected_firstnames, list):
        selected_firstnames = [selected_firstnames]
    if not isinstance(selected_genders, list):
        selected_genders = [selected_genders]
    
    filtered_data_subset = top_5[
        (top_5['Firstname'].isin(selected_firstnames)) & 
        (top_5['Gender'].isin(selected_genders))
    ]
    
    fig = go.Figure()

    # Iterate through each row in the filtered data subset
    for i, row in filtered_data_subset.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row[1:],  # Data for each religion for the current name
            theta=categories,
            fill='none',
            name=row['Firstname'] + " (" + row['Gender'] + ")",
        ))

    # Calculate the maximum value from the selected data
    if len(filtered_data_subset) > 0:  # Check if the DataFrame is not empty
        numeric_data = filtered_data_subset.select_dtypes(include=[np.number])  # Select only numeric columns
        max_value = np.max(numeric_data.values)
    else:
        max_value = 0  # Set max_value to 0 if the DataFrame is empty

    fig.update_layout(
        title=f'Top Five Names With The Greatest Overlap Between Religions',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 650]  # Set the range dynamically based on the max value
            )
        ),
        showlegend=True  # Set to True to display legend
    )

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=7530)


# In[ ]:


import pandas as pd

# Load the data
data3 = data.copy()

# Ensure 'SumYears' is treated as an integer column.
# If it's already an integer, no action is needed.
# If it's not, then we convert it to string, clean it, and then convert to integer.
if data3['SumYears'].dtype != 'int64':
    data3['SumYears'] = pd.to_numeric(data3['SumYears'].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)

# Calculating 'num_of_letters' based on the length of 'Firstname'
data3['num_of_letters'] = data3['Firstname'].str.len()

# Group by 'Religion' and 'num_of_letters', sum 'SumYears' for each group to get the total count
grouped_data_3 = data3.groupby(['Religion', 'num_of_letters'], as_index=False)['SumYears'].sum()

# Sort the grouped data by 'num_of_letters' and 'Religion' for better readability
grouped_data_3 = grouped_data_3.sort_values(by=['num_of_letters', 'Religion'])

# Display the grouped data
grouped_data_3.head(10)


# In[ ]:


#for dimesion of the spyder graph
categories = ['2', '3', '4', '5', '6', '7']

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.Label("Select Religion:"),
    dcc.Dropdown(
        id='Religion-dropdown',
        options=[{'label': Religion, 'value': Religion} for Religion in grouped_data_3['Religion'].unique()],
        value=[grouped_data_3['Religion'].iloc[0]],  # Set default value as a list
        multi=True
    ),
    dcc.Graph(id='spider-chart')  # Add an empty graph component to be updated by the callback
])

# Define callback to update the spider chart based on filter selections
@app.callback(
    Output('spider-chart', 'figure'),
    [Input('Religion-dropdown', 'value')]
)
def update_spider_chart(Select_Religion):
    fig = go.Figure()

    for religion in Select_Religion:
        religion_data = grouped_data_3[grouped_data_3['Religion'] == religion]
        values = religion_data['SumYears'].tolist()  # Use the 'Count' column for values
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='none',
            name=religion
        ))

    fig.update_layout(
        title=f'Letters Counts Per Name Across Religions',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, grouped_data_3['SumYears'].max()]  # Update the range based on the maximum value in 'Count'
            )
        ),
        showlegend=True
    )

    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=7540)


# In[ ]:


#for dimesion of the spyder graph
categories = ['2', '3', '4', '5', '6', '7']

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.Label("Select Religion:"),
    dcc.Dropdown(
        id='Religion-dropdown',
        options=[{'label': Religion, 'value': Religion} for Religion in grouped_data_3['Religion'].unique()],
        value=[grouped_data_3['Religion'].iloc[0]],  # Set default value as a list
        multi=True
    ),
    dcc.Graph(id='spider-chart')  # Add an empty graph component to be updated by the callback
])

# Define callback to update the spider chart based on filter selections
@app.callback(
    Output('spider-chart', 'figure'),
    [Input('Religion-dropdown', 'value')]
)
def update_spider_chart(Select_Religion):
    fig = go.Figure()

    for religion in Select_Religion:
        religion_data = grouped_data_3[grouped_data_3['Religion'] == religion]
        values = religion_data['SumYears'].tolist()  # Use the 'Count' column for values
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='none',
            name=religion,
            showlegend= True  # Remove the legend for this trace
        ))

    fig.update_layout(
        title=f'Letters Counts Per Name Across Religions (Log Scale)',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[np.log10(grouped_data_3['SumYears'].min()), np.log10(grouped_data_3['SumYears'].max())],  # Update the range based on the minimum and maximum value in 'Count'
                type='log',  # Set the radial axis type to log scale
                tickangle=0  # Align the tick labels horizontally
            )
        ),
        showlegend=True  # Keep the overall legend visible
    )

    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=7550)


# In[ ]:


#Names through war stories 
#we looke at the year of the war +1 after the war to see if a name gaind popularity

# Create a Dash application
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id='name-dropdown', #names we selected to present
        options=[
            {'label': 'גאיה', 'value': 'גאיה'},
            {'label': 'איתן', 'value': 'איתן'},
            {'label': 'מיכל', 'value': 'מיכל'},
            {'label': 'משה', 'value': 'משה'},
            
        ],
        value='גאיה'  # Default value
    ),
    dcc.Graph(id='name-graph')
])

@app.callback(
    Output('name-graph', 'figure'),
    [Input('name-dropdown', 'value')]
)
def update_graph(selected_name):
    #creating and updatin the grpah depends on user choice its limted by out chice we selected names with sotrys that we know
    fig = go.Figure()

    filtered_df2 = df_plotly[df_plotly['Firstname'].str.strip().str.lower() == selected_name.lower()]
    
    if not filtered_df2.empty:
        name_count = filtered_df2.iloc[:, 3:-2].sum(axis=0)
        years = df_plotly.columns[3:-3].astype(str)
        fig.add_trace(go.Scatter(x=years, y=name_count, mode='lines', name=selected_name, fill='none', line_color = "black", opacity=0.7, line=dict(width=7) ))
        
    else:
       
        fig.add_annotation(text='No data found for name: {}'.format(selected_name), xref='paper', yref='paper', showarrow=False, font=dict(size=12))

    fig.update_layout(
        title=f"Names through war stories: '{selected_name}'",
        xaxis_title='Year',
        yaxis=dict(title='Count of Name'),
        xaxis=dict(tickmode='linear', dtick=3),
        showlegend=False,
        xaxis_showgrid=False,  # Hide x-axis gridlines
        yaxis_showgrid=False,
        xaxis_zeroline=False,  # Hide the x-axis zero line
        yaxis_zeroline=False, 
    )
    
    if selected_name == "איתן":
        # Highlighting "צוק איתן" period
        fig.add_vrect(x0='2014', x1='2015', line_width=0, fillcolor="#03FF02", opacity=0.3)

        # Updating layout to focus on years 2010 to 2018
        fig.update_layout(
            xaxis=dict(
                type='linear',  # Use 'linear' since we're dealing with simple numeric values for years
                range=[2010, 2018]  # Setting the visible range from 2010 to 2021
            )
        )

        # Correcting the annotation to use an actual year value that matches the x-axis scale
        fig.add_annotation(
            x='2014.5',  
            y=0.75, 
            text="צוק איתן",
            showarrow=False,
            xref="x", 
            yref="paper",
            font=dict(size=20, color="black"),
            xanchor="center",
            yanchor="top"
        )
        
        story = ' מהמקום התשיעי ברשימת השמות העבריים ב2013 טיפס איתן עד למקום השני ב2015.<br> "הורים רבים מספרים שהם בחרו בשם למרות המלחמה השם איתן, המסמל חוזק וחוסן, ראה עלייה בפופולריות בסביבות מבצע צוק איתן בשנת 2014.<br> הזינוק הזה נבע ככל הנראה מתחושת הגאווה הלאומית והתכונות המעוררות הערצה שייצג המאמץ המלחמתי'
        
        fig.add_annotation( x='2014', text= story ,font=dict(size=10, color="black"), yref="paper" ,xanchor="center", showarrow=False)
        fig.add_trace(go.Scatter(x=years, y=name_count, mode='lines', name=selected_name, fill='none', line_color = "#03FF02", opacity=1))    

    elif selected_name == "גאיה":
        # Highlighting "המערכה ברצועת הבטחון" period
        fig.add_vrect(x0='1985', x1='2000', line_width=0, fillcolor="yellow", opacity=0.3)
        fig.add_vrect(x0='1994', x1='1994.2', line_width=0, fillcolor="gray", opacity=0.3)
        # Updating layout to focus on years 2010 to 2021
        fig.update_layout(
            xaxis=dict(
                type='linear',  # Use 'linear' since we're dealing with simple numeric values for years
                range=[1980, 2010]  # Setting the visible range from 2010 to 2021
            )
        )

        # Correcting the annotation to use an actual year value that matches the x-axis scale
        fig.add_annotation(
            x='1992',  
            y=1, 
            text="המערכה ברצועת הבטחון",
            showarrow=False,
            xref="x", 
            yref="paper",
            font=dict(size=20, color="black"),
            xanchor="center",
            yanchor="top"
        )
        
        story1 = "גאיה קורן הכינה סרט לזכרו של בן זוגה (רציתי להגיד לך)<br>, מפ צנחנים שנהרג בדרום לבנון. את מה שקרה אחר כך היא לא דמיינה <br> עשרות גאיות שנקראות על שמה בעקבות הסרט שנעשה לפני 29 שנה"
        fig.add_annotation( x='1992', text=story1 ,font=dict(size=12, color="black"), yref="paper" ,xanchor="center", showarrow=False)           
        fig.add_trace(go.Scatter(x=years, y=name_count, mode='lines', name=selected_name, fill='none', line_color = "yellow", opacity=1))

    elif selected_name == "מיכל":
        # Highlighting "מלחמת הכיפורים" period
        fig.add_vrect(x0='1973', x1='1974', line_width=0, fillcolor="blue", opacity=0.3)
        # Updating layout to focus on years 1960 to 1980
        fig.update_layout(
            xaxis=dict(
                type='linear',  # Use 'linear' since we're dealing with simple numeric values for years
                range=[1960, 1980]  # Setting the visible range from 2010 to 2021
            )
        )

        # Correcting the annotation to use an actual year value that matches the x-axis scale
        fig.add_annotation(
            x='1973.5',  
            y=0.75, 
            text="מלחמת יום הכיפורים",
            showarrow=False,
            xref="x", 
            yref="paper",
            font=dict(size=20, color="black"),
            xanchor="center",
            yanchor="top"
        )
        
        story2 = " בשנת 1973 השם הנפוץ ביותר היה מיכל <br> ככל הנראה בגלל הקרבה לראשי התיבות של (מ)לחמת (י)ום (כ)יפור (בתוספת האות ל')"
        fig.add_annotation( x='1973.5', text=story2 ,font=dict(size=12, color="black"), yref="paper" ,xanchor="center", showarrow=False)           
        fig.add_trace(go.Scatter(x=years, y=name_count, mode='lines', name=selected_name, fill='none', line_color = "#9067FF", opacity=1))

        
    elif selected_name == "משה":
        # Highlighting "ששת הימים" period
        fig.add_vrect(x0='1967', x1='1968', line_width=0, fillcolor="red", opacity=0.3)
        # Updating layout to focus on years 1960 to 1980
        fig.update_layout(
            xaxis=dict(
                type='linear',  # Use 'linear' since we're dealing with simple numeric values for years
                range=[1960, 1980]  # Setting the visible range from 2010 to 2021
            )
        )

        # Correcting the annotation to use an actual year value that matches the x-axis scale
        fig.add_annotation(
            x='1967.5',  
            y=0.75, 
            text="מלחמת ששת הימים",
            showarrow=False,
            xref="x", 
            yref="paper",
            font=dict(size=20, color="black"),
            xanchor="center",
            yanchor="top"
        )
        
        story3 = "בשנת 1967 השם הנפוץ ביותר היה משה' <br> ייתכן בשל העובדה שהוא מסמל את ראשי התיבות (מ)לחמת (ש)שת (ה)ימים שנערכה באותה השנה."
        fig.add_annotation( x='1967.5', text=story3 ,font=dict(size=12, color="black"), yref="paper" ,xanchor="center", showarrow=False, y=0.40)           
        fig.add_trace(go.Scatter(x=years, y=name_count, mode='lines', name=selected_name, fill='none', line_color = "#FF1A39", opacity=1))

    return fig

#Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port = 7560)
    


