from dash import dcc, html

def load_chatbot_layout(initial_chat_history):
    return html.Div(style={
        'backgroundColor': '#1C1C1C', # Dark background color from screenshot
        'minHeight': '80vh',        # Full viewport height
        'display': 'flex',
        'flexDirection': 'column',   # Main container is a flex column
        'fontFamily': 'Roboto, sans-serif',
        'color': "#FFFFFF",
        'overflow': 'hidden',        # Prevent the main page from scrolling
    }, children=[

        #Main centered content wrapper (for chat history)
        html.Div(style={
            'flexGrow': '1',
            'backgroundColor': "#1C1C1C",
            'display': 'flex',
            'width': '100%',
            'justifyContent': 'center',
            'paddingBottom': '100px', # Space for the fixed input area
            'boxSizing': 'border-box',
        }, children=[
            # Chat history display area - ONLY THIS DIV SCROLLS
            html.Div(
                id='chat-history-container',
                style={
                    'flexGrow': '1',
                    'maxWidth': '900px',
                    'width': '100%',
                    'overflowY': 'auto',
                    'padding': '20px',
                    'display': 'flex',
                    'flexDirection': 'column',
                    'gap': '15px',
                },
                children=[
                    # Initial messages will be rendered here by the callback on load
                    # The 'chat-end-marker' will always be the last element
                ]
            ),
        ]),

        # Input area FIXED at the bottom of the viewport
        html.Div(style={
            'padding': '15px 20px',
            'backgroundColor': '#282828',
            'borderTop': '1px solid #333',
            'position': 'fixed',
            'bottom': '0',
            'left': '50%',           # Center horizontally
            'transform': 'translateX(-50%)', # Adjust for element's own width
            'width': '50vw',          # Take up 50% of viewport width
            'maxWidth': '900px',      # Limit max width on very large screens
            'display': 'flex',
            'justifyContent': 'center', # This centers the *child* element within this bar
            'boxSizing': 'border-box',
            'zIndex': '1000'
        }, children=[
            html.Div(style={
                'display': 'flex',
                'alignItems': 'center',
                'backgroundColor': '#2E2E2E',
                'borderRadius': '20px',
                'border': '1px solid #555',
                'padding': '8px 15px',
                'maxWidth': '900px',
                'width': '100%'
            }, children=[
                dcc.Textarea(
                    id='user-input',
                    placeholder='Your message...',
                    rows=1,
                    style={
                        'flexGrow': '1',
                        'backgroundColor': 'transparent',
                        'border': 'none',
                        'color': '#E0E0E0',
                        'fontSize': '16px',
                        'resize': 'none',
                        'outline': 'none',
                        'minHeight': '20px',
                        'padding': '0',
                        'margin': '0'
                    }
                ),
                html.Button(
                    html.I(className="fas fa-paper-plane"),
                    id='send-button',
                    n_clicks=0,
                    style={
                        'backgroundColor': 'transparent',
                        'border': 'none',
                        'color': '#E0E0E0',
                        'fontSize': '18px',
                        'cursor': 'pointer',
                        'marginLeft': '10px',
                        'padding': '5px',
                        'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'
                    }
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
        return html.Div(style={
            'display': 'flex', 'justifyContent': 'flex-end', 'marginBottom': '10px'
        }, children=[
            html.Div(style={
                'backgroundColor': '#0056b3', 'borderRadius': '8px', 'padding': '10px 15px',
                'maxWidth': '70%', 'wordBreak': 'break-word', 'border': '1px solid #004085',
            }, children=[
                html.Span("User:", style={'fontWeight': 'bold', 'color': '#FFFFFF', 'marginRight': '5px'}),
                dcc.Markdown(message_content, style={'margin': '0', 'whiteSpace': 'pre-wrap', 'color': '#FFFFFF'})
            ])
        ])
    else: # sender == "bot"
        return html.Div(style={
            'display': 'flex', 'justifyContent': 'flex-start', 'marginBottom': '10px'
        }, children=[
            html.Div(style={
                    'backgroundColor': '#3A3A3A', 'borderRadius': '8px', 'padding': '10px 15px',
                    'maxWidth': '70%', 'wordBreak': 'break-word', 'border': '1px solid #4A4A4A',
                }, children=[
                    html.Span("Bot:", style={'fontWeight': 'bold', 'color': '#00BFFF', 'marginRight': '5px'}),
                    dcc.Markdown(message_content, style={'margin': '0', 'whiteSpace': 'pre-wrap', 'color': '#E0E0E0'})
                ]
            )
        ])