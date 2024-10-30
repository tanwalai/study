import pandas as pd
import re
import json
import requests
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import plotly.express as px
import base64
import io
import time

# URLs for the primary and backup APIs
PRIMARY_API_URL = 'https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1'
BACKUP_API_URL = 'https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2'

# Token for authorization
token = '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'
headers = {'Authorization': f'Bearer {token}'}

def query(payload, max_chars_per_request=1000, use_backup=False):
    text = payload['inputs']
    num_chunks = (len(text) + max_chars_per_request - 1) // max_chars_per_request
    
    responses = []
    for i in range(num_chunks):
        start = i * max_chars_per_request
        end = min((i + 1) * max_chars_per_request, len(text))
        chunk_payload = {'inputs': text[start:end]}

        # Choose the API URL based on whether backup should be used
        api_url = BACKUP_API_URL if use_backup else PRIMARY_API_URL
        
        try:
            response = requests.post(api_url, headers=headers, json=chunk_payload)
            response.raise_for_status()  # Will raise an exception for 4XX or 5XX errors
            response_text = response.json()[0]['generated_text']
            responses.append(response_text)
        except requests.exceptions.RequestException as e:
            # If error occurs and backup has not been tried yet, switch to backup API
            if not use_backup:
                print("Primary API failed, switching to backup API.")
                return query(payload, max_chars_per_request, use_backup=True)
            else:
                # If the backup also fails, re-raise the exception
                raise Exception("Both primary and backup APIs failed.") from e
    
    return ''.join(responses)

def format_instruction(prompt:str,
                       instruction: str):
    instruction_prompt = prompt
    instruction_text = "[INST] " + instruction_prompt + instruction + " [/INST]"
    return instruction_text

def format_output(output, instruction):
    output = output.replace(instruction, '').strip()
    return output

def generate_output(instruction: str,
                    prompt:str):
    instruction = format_instruction(instruction = instruction, prompt = prompt)
    data = query({"inputs": instruction,
                  "parameters" : {"max_length": 10000}})
    output = format_output(output = data, instruction = instruction)
    return output

def get_chart_prompt():
    global df
    chart_prompt = f"""
    My data contains columns: {', '.join(df.columns)}.
    Chart options: Bar, Scatter, Pie, Line, Box plot 
    Please shortly suggest chart type and columns (based on column names provided) needed for the following question:
    
    """
    return chart_prompt

def get_column_needed(df:pd.DataFrame, generated_text:str):
    used_col = [col for col in df.columns if col.lower() in generated_text.replace('\\','').lower()]
    return used_col

def get_axis_prompt():
    axis_prompt = f"""
    Return me this form {{"dimension": ["xxx"], "metrics": ["yyy"]}} from the column lists:
    """
    return axis_prompt

def extract_dimension_metrics(generated_text:str):
    # pattern = r'"dimension"\s*:\s*\["(.*?)"\]'
    dimension_pattern_1 = r'"dimension"\s*:\s*(\["([^xxx"]*)"])'
    dimension_pattern_2 = r'"dimension"\s*:\s*(\[.*?\])'
    metrics_pattern_1 = r'"metrics"\s*:\s*(\["([^yyy"]*)"])'
    metrics_pattern_2 = r'"metrics"\s*:\s*(\[.*?\])'

    dimension_match_1 = re.search(dimension_pattern_1, generated_text)
    dimension_match_2 = re.search(dimension_pattern_2, generated_text)
    metrics_match_1 = re.search(metrics_pattern_1, generated_text)
    metrics_match_2 = re.search(metrics_pattern_2, generated_text)

    if dimension_match_1:
        x = dimension_match_1.group(1).replace('\\','')
        x = json.loads(x)
    elif dimension_match_2:
        x = dimension_match_2.group(1).replace('\\','')
        x = json.loads(x)
    else:
        x = []
    if metrics_match_1:
        y = metrics_match_1.group(1).replace('\\','')
        y = json.loads(y)
    elif metrics_match_2:
        y = metrics_match_2.group(1).replace('\\','')
        y = json.loads(y)
    else:
        y = []

    try:
        x.remove('xxx')
    except:
        pass
    try:
        y.remove('yyy')
    except:
        pass
    
    return x, y


def get_chart_axis(df:pd.DataFrame, column:list, x, y):
    # x = []
    # y = []

    if x == [] and y ==[]:
        for col in column:
            if str(df[col].dtype) in ['object', 'str', 'string']:
                x.append(col)
            elif col.lower() in ['date', 'year', 'month', 'week']:
                x.append(col)
            elif y == []:
                y.append(col)
            else:
                x.append(col)
    elif x == [] and len(y) > 1:
        for col in column:
            if str(df[col].dtype) in ['object', 'str', 'string']:
                x.append(col)
                y.remove(col)
            elif col.lower() in ['date', 'year', 'month', 'week']:
                x.append(col)
                y.remove(col)
            else:
                x.append(col)
    elif y == [] and len(x) > 1:
        for col in column:
            if str(df[col].dtype) not in ['object', 'str', 'string'] and col.lower() in ['date', 'year', 'month', 'week']:
                y.append(col)
                x.remove(col)

    return x, y


def suggest_chart_type(df:pd.DataFrame, generated_text:str):
    if 'scatter' in generated_text.lower():
        return 'scatter'
    elif 'pie' in generated_text.lower():
        return 'pie'
    elif 'line' in generated_text.lower():
        return 'line'
    elif 'bar' in generated_text.lower():
        return 'bar'
    elif 'box' in generated_text.lower():
        return 'box'
    else:
        return 'bar'
    
def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'csv' in content_type:
        # Decode CSV
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    elif 'xls' in content_type:
        # Decode Excel
        df = pd.read_excel(io.StringIO(decoded.decode('utf-8')))
        # decoded = pd.read_excel(io.BytesIO(content_string))

    return df

def transform_axis(df,input_str, x, y):
    if 'average' in input_str.lower() or 'mean' in input_str.lower():
        df2 = df.groupby(x).mean(y).reset_index()
    elif 'total' in input_str.lower() or 'sum' in input_str.lower():
        df2 = df.groupby(x).sum(y).reset_index()
    else:
        df2 = df.groupby(x).sum(y).reset_index()
    return df2

def pie_chart(df:pd.DataFrame,
              x:list,
              y:list):
              
    fig = px.pie(df, 
                 names = x[0],
                 values = y[0],
                 color_discrete_sequence = px.colors.qualitative.Pastel,
                 hole = 0.4)
    return fig

def bar_chart(df:pd.DataFrame,
              x:list,
              y:list):
    bar_df = df.copy()
    bar_df[x[0]] = bar_df[x[0]].astype('object')
    fig = px.bar(df, 
                 x = x[0], 
                 y = y[0],
                 color = x[0],
                 color_discrete_sequence = px.colors.qualitative.Pastel
                 )
    return fig

def line_chart(df:pd.DataFrame,
               x:list,
               y:list):
    fig = px.line(df, 
                  x = x[0], 
                  y = y[0],
                  color = x[0],
                  color_discrete_sequence = px.colors.qualitative.Pastel
                  )
    return fig

def scatter_chart(df:pd.DataFrame,
                  x:list,
                  y:list):
    scatter_df = df.copy()
    scatter_df[y].fillna(0, inplace = True)
    scatter_df[x[0]].fillna('NA', inplace = True)
    try:
        scatter_df[x[1]].fillna('NA', inplace = True)
    except:
        pass
    
    if len(x) == 1:
        fig = px.scatter(scatter_df, 
                         x = x[0], 
                         y = y[0],
                         color = x[0],
                         color_discrete_sequence = px.colors.qualitative.Pastel
                         )
    elif scatter_df[x[0]].nunique() > scatter_df[x[1]].nunique():
        fig = px.scatter(scatter_df, 
                         x = x[0], 
                         y = y[0],
                         color_discrete_sequence = px.colors.qualitative.Pastel
                         )
    else:
        fig = px.scatter(scatter_df, 
                         x = x[1], 
                         y = y[0],
                         color_discrete_sequence = px.colors.qualitative.Pastel
                         )
    return fig

def box_plot(df:pd.DataFrame,
             y:list):
    fig = px.box(df, 
                 y = y,
                 color_discrete_sequence = px.colors.qualitative.Pastel
                 )
    return fig

def table_chart(df):
    fig = go.Figure(
        data = [
            go.Table(
                header = dict(values = list(df.columns),
                              fill_color = 'lightblue',
                              line_color = 'black',
                              align = 'left'),
                cells = dict(values = [df[col] for col in df.columns],
                             fill_color = 'white',
                             line_color = 'black',
                             align = 'left')
            )
        ]
    )
    return fig

def generate_chart(df,chart_type, x, y):
    # filtered_data = df2[df2[chart_json['filter']['column']].isin(chart_json['filter']['value'])]
    filtered_data = df.copy()
    filtered_data = filtered_data.sort_values(by=x)

    if chart_type == 'pie':
        fig = pie_chart(df = filtered_data,
                        x = x,
                        y = y)
    elif chart_type == 'line':
        fig = line_chart(df = filtered_data,
                         x = x,
                         y = y)
    elif chart_type == 'bar':
        fig = bar_chart(df = filtered_data,
                        x = x,
                        y = y)
    elif chart_type == 'scatter':
        fig = scatter_chart(df = filtered_data,
                            x = x,
                            y = y)
    elif chart_type == 'box':
        fig = box_plot(df = filtered_data,
                       y = y)
    else:
        fig = table_chart(df = filtered_data)

    return fig

app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])
fig = go.Figure()

custom_css = {
    'fontFamily': 'Open Sans, sans-serif'
}

sidebar = html.Div(
    [
        # Header
        dbc.Row(
            [
                html.H5(
                    'DADS5001', 
                    style = {'margin-top': 'auto',
                             'margin-bottom': 'auto',
                             'margin-left': 'auto',
                             'margin-right': 'auto', 
                             'width': '100%',
                             **custom_css}
                )
            ],
            style = {"height": "10vh"}
        ),

        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            dcc.Upload(
                                id = 'upload-data',
                                children=html.Div(
                                    [
                                        'Drag and Drop or ',
                                        html.A('Select Files')
                                    ],
                                    style = {'width': '100%'}
                                ),
                                style = {'width': '100%',
                                         'height': '60px',
                                         'lineHeight': '60px',
                                         'borderWidth': '1px',
                                         'borderStyle': 'dashed',
                                         'borderRadius': '5px',
                                         'textAlign': 'center',
                                         'margin': '10px'
                                },
                                multiple = False
                            ),
                            html.Div(id = 'output-data-upload')
                        ]
                    )
                )
            ]
        )

        # # Filter 1
        # dbc.Row(
        #     [
        #         dbc.Col(
        #             html.Div(
        #                 [
        #                     html.P(
        #                         'Filter Field', 
        #                         style = {'margin-top': '8px', 
        #                                  'margin-bottom': '4px',
        #                                  **custom_css}, 
        #                         className = 'font-weight-bold',
        #                     ),
        #                     dcc.Dropdown(
        #                         id = 'filter-1',
        #                         multi = False,
        #                         # options = [{'label': x, 'value': x} for x in ['option1','option2']],
        #                         style = {'width': '100%'}
        #                     )
        #                 ]
        #             )
        #         )
        #     ]
        # ),

        # Filter 2
        # dbc.Row(
        #     [
        #         dbc.Col(
        #             html.Div(
        #                 [
        #                     html.P(
        #                         'Filter Value', 
        #                         style = {'margin-top': '16px', 'margin-bottom': '4px'}, 
        #                         className = 'font-weight-bold'
        #                     ),
        #                     dcc.Dropdown(
        #                         id = 'filter-2',
        #                         multi = True,
        #                         # options = [{'label': x, 'value': x} for x in ['option1','option2']],
        #                         style = {'width': '100%'}
        #                     ),
        #                     # html.Button(
        #                     #     id = 'my-button', 
        #                     #     n_clicks = 0, 
        #                     #     children = 'Apply',
        #                         #     style = {'width': '100%',
        #                         #              'height': '5vh',
        #                         #              'margin-top': '25px',
        #                         #              'margin-bottom': '6px',
        #                         #              'border': '1px',
        #                         #              'border-radius': '8px'},
        #                     #     className = 'bg-primary text-white font-italic'),
        #                     html.Hr()
        #                 ]
        #             )
        #         )
        #     ]
        # )

        # Filter 3
    ],
    style={'height': '100vh', 'border-radius': '15px'}
)

content = html.Div(
    [
        # Header
        dbc.Row(
            dbc.Col(
                html.H1(
                    'Generative AI Dashboard', 
                    style={'color': 'black', 'margin-top': '10px', 'margin-left': '20px',
                           'margin-bottom': 'auto', 'height': '5px', 'font-size': '20px'}
                ),
                width=7,
                style={'padding': '15px', 'marginBottom': '20px'}
            )
        ),

        # Chart and Description
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        id="dynamic-plot",
                        figure=fig,
                        style={'width': '100%', 'height': '100%', 'padding': '0px'},
                        className='bg-light'
                    ),
                    style={'margin-left': '0px', 'border': '0 px solid lightgrey', 'border-radius': '5px'},
                    width=8
                ),
                dbc.Col(
                    html.Div(
                        id="chart-description",
                        style={'width': '100%', 'height': '100%', 'padding': '0px'},
                    ),
                    style={'display': 'flex','flex-direction': 'column','justify-content': 'center','margin-right': '0.1px', 'border': '0 px solid lightgrey', 'border-radius': '5px','font-size': '16px'},
                    width=3
                )
            ],
            style={'height': '70vh'}
        ),

        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        dcc.Input(
                            id='text-input',
                            type='text',
                            placeholder='Enter text here...',
                            style={'border-radius': '10px', 'margin-left': '15px', 'width': '85%', **custom_css}
                        ),
                        html.Button(
                            'Send', 
                            id='my-button', 
                            n_clicks=0,
                            style={'border-radius': '8px', 'margin-left': '3px', 'width': '80px'},
                            className='btn btn-primary'
                        )
                    ],
                    style={'padding': '20px', 'marginTop': '10px'}
                ),
                width=12
            )
        )
    ],
    style={'margin-top': '20px', 'margin-bottom': '0px', 'margin-left': '5px', 'margin-right': '10px',
           'backgroundColor': '#FFFFFF', 'height': '95vh', 'border-radius': '25px'}
)

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    sidebar,
                    width = 2,
                    style = {'backgroundColor': '#EDF2FA'}
                ),
                dbc.Col(content, width=10)
            ],
            style = {"height": "100vh"}
        ),
    ],
    fluid = True,
    style = {'backgroundColor': '#EDF2FA', 
             'border-radius': '15px', 
             **custom_css}
)

# File upload
@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents')
)
def update_output(contents):
    if contents is None:
        raise PreventUpdate
    
    global df
    try:
        df = parse_contents(contents)
        print(df.head())
    except Exception as e:
        return html.Div([
            "There was an error processing the file: ",
            html.Strong(str(e))
        ], style={'color': 'red', 'marginTop': '20px'})

    dimension_col = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]

    return html.Div(
        [
            html.H6('Columns:'),
            html.P(', '.join(df.columns)),
        ],
        style={'marginTop': '20px'}
    )

# Update chart
@app.callback(
    [Output('dynamic-plot', 'figure'),
     Output('chart-description', 'children')],  # Fixed typo here
    Input('my-button', 'n_clicks'),
    State('text-input', 'value')
)
def update_dynamic_plot(n_clicks, input_text):
    global df
    start_time = time.time()
    if n_clicks > 0:
        # Generate output
        output = generate_output(instruction=input_text, prompt=get_chart_prompt())
        pre_description = generate_output(instruction=output, prompt=": this is the data about the chart, so please give description of the chart (the length of the full description that you give pls limit to 1,000 characters but don't have to return number of charater to me)")
        if  len(pre_description.split("creating the chart.")) > 1:
            description = pre_description.split("creating the chart.")[1]
        else : description = pre_description
        print(f'Output: {output}')
        print(df.columns)

        used_col = get_column_needed(df, generated_text=input_text)
        print(f'Columns: {used_col}')

        dimension_metrics_text = generate_output(instruction=', '.join(used_col), prompt=get_axis_prompt())
        print(f'Dimension Metrics Text: {dimension_metrics_text}')

        dimension, metrics = extract_dimension_metrics(generated_text=dimension_metrics_text)
        print("Initial dimensions:", dimension)
        print("Initial metrics:", metrics)
        
        if len(dimension) > 1:
            metrics.append(dimension[0])
            dimension.pop(0)
        print(f'Dimension : {dimension}')
        print(f'Metric : {metrics}')

        df2 = transform_axis(df,input_text,x=dimension[0],y=metrics[0])

        x, y = get_chart_axis(df2, column=used_col, x=dimension, y=metrics)
        print(f'X = {x} | Type: {type(x)}')
        print(f'Y = {y} | Type: {type(y)}')
        chart_type = suggest_chart_type(df, generated_text=output)
        print(f'Chart Type = {chart_type}')

        # Generate chart
        try:
            fig = generate_chart(df2,chart_type=chart_type, x=x, y=y)
            print('Plot Success')
        except Exception as e:
            print('Plot Failed:', str(e))
            fig = generate_chart(chart_type='table', x=x, y=y)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")

        return fig, description
    else:
        # Return an empty figure and a message asking for input
        return go.Figure(), "Please enter an instruction and click the send button."
    

if __name__ == '__main__':
    app.run_server(debug=True)