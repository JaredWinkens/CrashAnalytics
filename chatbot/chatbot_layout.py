from dash import dcc, html

def load_chatbot_layout(initial_chat_history):
    return html.Div(className='app-container', children=[
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
                dcc.Textarea(
                    id='user-input',
                    placeholder='Your message...',
                    rows=1,
                ),
                html.Button(
                    html.I(className="fas fa-paper-plane"),
                    id='send-button',
                    n_clicks=0,
                    title ="Send message"
                ),
                html.Button(
                    html.I(className="fas fa-trash"), # You can choose a different icon
                    id='clear-button',
                    n_clicks=0,
                    title ="Clear chat history",
                    className='clear-button' # Add a class for styling
                    
                )
            ])
        ]),

        # Hidden Div to store the chat history data
        dcc.Store(id='chat-history-store', data=initial_chat_history),

        # Hidden Div to trigger clientside scroll callback
        dcc.Store(id='scroll-trigger', data=0), # Data will increment to trigger callback

        # Store to hold the user's last question to trigger bot response
        dcc.Store(id='user-question-for-bot', data=None) # Will store {"question": "...", "timestamp": "..."}

    ])

# Function to render a single message bubble
def render_message_bubble(sender, message_content):
    if sender == "user":
        return html.Div(className='message-bubble-wrapper user-bubble-wrapper', children=[
            html.Div(className='message-bubble user-bubble', children=[
                dcc.Markdown(message_content, className='message-content'),
            ])
        ])
    else: # sender == "bot"
        return html.Div(className='message-bubble-wrapper bot-bubble-wrapper', children=[
            html.Div(className='message-bubble bot-bubble', children=[
                html.I(className="fas fa-robot sender-icon bot-sender-icon"),
                html.I(className="fa fa-info-circle", title ="Data is only available from 2020-2023", id='info-button'),
                dcc.Markdown(message_content, className='message-content'),
                
            ])
        ])