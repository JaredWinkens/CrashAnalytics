from dash import dcc, html
import dash_bootstrap_components as dbc

def load_chatbot_layout(initial_chat_history):
    return html.Div(className='app-container', children=[
        # Header Section
        html.Div([
            html.Div([
                html.Img(src='/assets/Poly.svg', style={
                    'height': '128px', 'float': 'left', 'margin-right': '40px', 
                    'margin-left': '-20px', 'margin-top': '-8px'
                }),
                html.H1('Safety Chatbot', className='app-title'),
                html.Img(src='/assets/NY.svg', className='ny-logo')
            ],style={
                'backgroundColor': '#18468B', 'padding': '7.5px', 'position': 'fixed', 
                'top': '50px', 'left': '0', 'width': '100%', 'zIndex': '999', 'height': '90px'
            }),
        ]),
        #Main centered content wrapper (for chat history)
        html.Div(className='main-content-wrapper', children=[
            # Chat history display area - ONLY THIS DIV SCROLLS
            html.Div(id='chat-history-container', children=[
                    # Initial messages will be rendered here by the callback on load
                    # The 'chat-end-marker' will always be the last element
                ]
            ),
        ]),

        # Input area FIXED at the bottom of the viewport
        html.Div(className='input-bar-fixed-wrapper', children=[
            html.Div(className='input-field-wrapper', children=[
                # dcc.Textarea(
                #     id='user-input',
                #     placeholder='Your message...',
                #     rows=1,
                # ),
                dcc.Input(
                    id='user-input',
                    placeholder='Your message...', 
                    type='text', 
                    value=''
                ),
                html.Button(
                    html.I(className="fas fa-paper-plane"),
                    id='send-button',
                    n_clicks=0,
                    title ="Send message"
                ),
            ]),
            html.Button(
                    html.I(className="fas fa-trash"), # You can choose a different icon
                    id='clear-button',
                    n_clicks=0,
                    title ="Clear chat history",
                    className='clear-button' # Add a class for styling 
                )
        ]),

        # Hidden Div to store the chat history data
        dcc.Store(id='chat-history-store', data=initial_chat_history),

        # Hidden Div to trigger clientside scroll callback
        dcc.Store(id='scroll-trigger', data=0), # Data will increment to trigger callback

        # Store to hold the user's last question to trigger bot response
        dcc.Store(id='user-question-for-bot', data=None) # Will store {"question": "...", "timestamp": "..."}

    ])

# Function to render a single message bubble
def render_user_message_bubble(message_content):
    return html.Div(className='message-bubble-wrapper user-bubble-wrapper', children=[
        html.Div(className='message-bubble user-bubble', children=[
            dcc.Markdown(message_content, className='message-content'),
        ])
    ])
    

def render_bot_message_bubble(message_content, fig, loading):
    if loading:
        return html.Div(className='message-bubble-wrapper bot-bubble-wrapper', children=[
            html.Div(className='message-bubble bot-bubble', children=[
                html.I(className="fas fa-robot sender-icon bot-sender-icon"),
                #html.I(className="fa fa-info-circle", title ="Data is only available from 2020-2023", id='info-button'),
                dcc.Markdown("", className='message-content'),
                html.Img(id='bot-loading-gif', src='assets/Loading-Dots-Blue-Cropped.gif')
            ])
        ])
    
    else:
        if fig == None:
            return html.Div(className='message-bubble-wrapper bot-bubble-wrapper', children=[
                html.Div(className='message-bubble bot-bubble', children=[
                    html.I(className="fas fa-robot sender-icon bot-sender-icon"),
                    html.I(className="fa fa-info-circle", title ="Data is only available from 2020-2023", id='info-button'),
                    dcc.Markdown(message_content, className='message-content'),
                    
                ])
            ])
        elif fig != None:        
            return html.Div(className='message-bubble-wrapper bot-bubble-wrapper', children=[
                html.Div(className='message-bubble bot-bubble', children=[
                    html.I(className="fas fa-robot sender-icon bot-sender-icon"),
                    html.I(className="fa fa-info-circle", title ="Data is only available from 2020-2023", id='info-button'),
                    dcc.Markdown(message_content, className='message-content'),
                    dcc.Graph(id='crash-map', figure=fig)
                ])
            ])