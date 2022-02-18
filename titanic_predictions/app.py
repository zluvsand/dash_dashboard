import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import plotly.graph_objects as go
import plotly.express as px

import dash
from dash import html, dcc
import dash_daq as daq
from dash.dependencies import Input, Output, State

# ********************* DATA PREPARATION *********************
# Load data
df = sns.load_dataset('titanic').drop(columns=['pclass', 'embarked', 'alive'])

# Format data for dashboard
df.columns = df.columns.str.capitalize().str.replace('_', ' ')
df.rename(columns={'Sex': 'Gender'}, inplace=True)
for col in df.select_dtypes('object').columns:
    df[col] = df[col].str.capitalize()

# Partition into train and test splits
TARGET = 'Survived'
y = df[TARGET]
X = df.drop(columns=TARGET)

numerical = X.select_dtypes(include=['number', 'boolean']).columns
categorical = X.select_dtypes(exclude=['number', 'boolean']).columns
X[categorical] = X[categorical].astype('object')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

# Build pipeline
pipeline = Pipeline([
    ('preprocessor', ColumnTransformer(transformers=[
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('encoder', OneHotEncoder(sparse=False))
            
        ]), categorical),
        ('num', SimpleImputer(strategy='mean'), numerical)
    ])),
    ('model', RandomForestClassifier(random_state=42))
])
pipeline.fit(X_train, y_train)

# Add predicted probabilities
test['Probability'] = pipeline.predict_proba(X_test)[:,1]
test['Target'] = test[TARGET]
test[TARGET] = test[TARGET].map({0: 'No', 1: 'Yes'})

labels = []
for i, x in enumerate(np.arange(0, 101, 10)):
    if i>0:
        labels.append(f"{previous_x}% to <{x}%")
    previous_x = x
test['Binned probability'] = pd.cut(test['Probability'], len(labels), labels=labels, right=False)

# Helper functions for dropdowns and slider
def create_dropdown_options(series):
    options = [{'label': i, 'value': i} for i in series.sort_values().unique()]
    return options
def create_dropdown_value(series):
    value = series.sort_values().unique().tolist()
    return value
def create_slider_marks(values):
    marks = {i: {'label': str(i)} for i in values}
    return marks

# ********************* DASH APP *********************
app = dash.Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.H1("Titanic predictions"),
        html.P("Summary of predicted probabilities for Titanic test dataset."),
        # html.Img(src=app.get_asset_url("left_pane.png")),
        html.Img(src="assets/left_pane.png"),
        html.Label("Passenger class", className='dropdown-labels'), 
        dcc.Dropdown(id='class-dropdown', className='dropdown', multi=True,
                     options=create_dropdown_options(test['Class']),
                     value=create_dropdown_value(test['Class'])),
        html.Br(),
        html.Label("Gender", className='dropdown-labels'), 
        dcc.Dropdown(id='gender-dropdown', className='dropdown', multi=True,
                     options=create_dropdown_options(test['Gender']),
                     value=create_dropdown_value(test['Gender'])),
        html.Button(id='update-button', children="Update", n_clicks=0),
        ], id='left-container'),
    html.Div([
        html.Div([
            dcc.Graph(id="histogram"),
            dcc.Graph(id="barplot")
        ], id='visualisation'),
        html.Div([
            dcc.Graph(id='table'),
            html.Div([
                html.Br(),
                html.Label("Survival status", className='other-labels'), 
                daq.BooleanSwitch(id='target_toggle', className='toggle', color="#FFBD59", on=True),
                html.Br(),
                html.Label("Sort probability in ascending order", className='other-labels'),
                daq.BooleanSwitch(id='sort_toggle', className='toggle', color="#FFBD59", on=True),
                html.Br(),
                html.Label("Number of records", className='other-labels'), 
                dcc.Slider(id='n-slider', min=5, max=20, step=1, value=10, 
                           marks=create_slider_marks([5, 10, 15, 20])),
                html.Br()
            ], id='table-side'),
        ], id='data-extract')
   ], id='right-container')
], id='container')

@app.callback(
    [Output(component_id='histogram', component_property='figure'),
     Output(component_id='barplot', component_property='figure'),
     Output(component_id='table', component_property='figure')],
    [State(component_id='class-dropdown', component_property='value'),
     State(component_id='gender-dropdown', component_property='value'),
     Input(component_id='update-button', component_property='n_clicks'),
     Input(component_id='target_toggle', component_property='on'),
     Input(component_id='sort_toggle', component_property='on'),
     Input(component_id='n-slider', component_property='value')]
)
def update_output(class_value, gender_value, n_clicks, target, ascending, n):
    dff = test.copy()
    
    if n_clicks>0:
        if len(class_value)>0:
            dff = dff[dff['Class'].isin(class_value)]
        elif len(class_value)==0:
            raise dash.exceptions.PreventUpdate
        
        if len(gender_value)>0:
            dff = dff[dff['Gender'].isin(gender_value)]
        elif len(gender_value)==0:
            raise dash.exceptions.PreventUpdate
    
    # Visual 1: Histogram
    histogram = px.histogram(dff, x='Probability', color=TARGET, opacity=0.6, marginal="box", 
                             color_discrete_sequence=['#FFBD59', '#3BA27A'], nbins=30)
    histogram.update_layout(title_text=f'Distribution of probabilities by class (n={len(dff)})',
                            font_family='Tahoma', plot_bgcolor='rgba(255,242,204,100)')
                            # paper_bgcolor='rgba(0,0,0,0)',
                            
    histogram.update_yaxes(title_text="Count")

    # Visual 2: Barplot
    barplot = px.bar(dff.groupby('Binned probability', as_index=False)['Target'].mean(), 
                     x='Binned probability', y='Target', color_discrete_sequence=['#3BA27A'])
    barplot.update_layout(title_text=f'Survival rate by binned probabilities (n={len(dff)})', 
                          font_family='Tahoma', xaxis = {'categoryarray': labels}, 
                          plot_bgcolor='rgba(255,242,204,100)')
    barplot.update_yaxes(title_text="Percentage survived")

    # Visual 3: Table
    if target==True:
        dff = dff[dff['Target']==1]
    else:
        dff = dff[dff['Target']==0]
          
    dff = dff.sort_values('Probability', ascending=ascending).head(n)
    
    columns = ['Age', 'Gender', 'Class', 'Embark town', TARGET, 'Probability']
    table = go.Figure(data=[go.Table(
        header=dict(values=columns, fill_color='#FFBD59', line_color='white',
                    font=dict(color='white', size=13), align='center'),
        cells=dict(values=[dff[c] for c in columns], format=["d", "", "", "", "", ".2%"],
                   fill_color=[['white', '#FFF2CC']*(len(dff)-1)], align='center'))
    ])
    table.update_layout(title_text=f'Sample records (n={len(dff)})', font_family='Tahoma')

    return histogram, barplot, table

if __name__ == '__main__':
    app.run_server(debug=True)