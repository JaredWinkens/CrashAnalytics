from app_instance import app
from dash import dcc, html, clientside_callback, Input, Output, State
from layouts import chatbot_layout
from chatbot import chatbot
import dash
import datetime

@app.callback(
    Output('chat-history-store', 'data'),
    Output('chat-history-container', 'children'),
    Input('clear-button', 'n_clicks'),
    State('chat-history-store', 'data'),
    State('chat-history-container', 'children'),
    prevent_initial_call=True
)
def clear_chat_history(n_clicks, current_chat_data, curent_chat_container):
    if n_clicks and n_clicks > 0:
        chatbot.create_new_chat_session()
        return current_chat_data[0:1], curent_chat_container[0:1]
    return dash.no_update, dash.no_update

# --- Handle User Input and Display Immediately (with loading placeholder) ---
@app.callback(
    Output('user-input', 'value'),
    Output('chat-history-store', 'data', allow_duplicate=True),
    Output('scroll-trigger', 'data', allow_duplicate=True),
    Output('user-question-for-bot', 'data'),
    [Input('send-button', 'n_clicks'),
     Input('user-input', 'n_submit')],
    State('user-input', 'value'),
    State('chat-history-store', 'data'),
    State('scroll-trigger', 'data'),
    prevent_initial_call=True
)
def handle_user_input(send_button_clicks, n_submits, user_question, current_chat_data, current_scroll_trigger):
    if not user_question or user_question.strip() == "":
        raise dash.exceptions.PreventUpdate

    # Append user message
    msg = {"sender": "user", "message": user_question}
    current_chat_data.append(msg)

    # Append temporary loading message
    loading_msg = {"sender": "bot", "message": "Thinking...", "map": None, "loading": True}
    current_chat_data.append(loading_msg)

    new_scroll_trigger = current_scroll_trigger + 1

    return (
        '',
        current_chat_data, 
        new_scroll_trigger, 
        {
            "question": user_question,
            "timestamp": datetime.datetime.now().isoformat(),
        }
    )

# --- Generate Bot Response (updates the specific bot message) ---
@app.callback(
    Output('chat-history-store', 'data', allow_duplicate=True),
    Output('scroll-trigger', 'data', allow_duplicate=True),    
    Input('user-question-for-bot', 'data'),
    State('chat-history-store', 'data'),
    State('scroll-trigger', 'data'),
    prevent_initial_call=True
)
def generate_and_display_bot_response(user_question_data, current_chat_data, current_scroll_trigger):
    if user_question_data is None:
        raise dash.exceptions.PreventUpdate

    user_question = user_question_data["question"]

    bot_response_data = chatbot.get_agent_response(user_question)
    bot_response_text = bot_response_data.get("text", "No response.")
    fig = bot_response_data.get("visualization_data")
    
    # Remove loading message
    current_chat_data.pop()

    msg = {"sender": "bot", "message": bot_response_text, "map": fig, "loading": False}
    current_chat_data.append(msg)
    new_scroll_trigger = current_scroll_trigger + 1

    return current_chat_data, new_scroll_trigger

# --- Update Chat History Display and Scroll after all data is in chat-history-store ---
@app.callback(
    Output('chat-history-container', 'children', allow_duplicate=True),
    Output('chat-history-store', 'data', allow_duplicate=True),
    Output('scroll-trigger', 'data', allow_duplicate=True),
    Input('chat-history-store', 'data'),
    prevent_initial_call='initial_duplicate'
)
def update_chat_display(stored_chat_data):
    if stored_chat_data is None:
        raise dash.exceptions.PreventUpdate
    
    rendered_history_elements =[]
    for msg in stored_chat_data:
        if msg['sender'] == "user":
            rendered_history_elements.append(chatbot_layout.render_user_message_bubble(msg['message']))
        elif msg['sender'] == "bot":
            rendered_history_elements.append(chatbot_layout.render_bot_message_bubble(msg['message'], msg['map'], msg['loading']))
    rendered_history_elements.append(html.Div(id='chat-end-marker'))
    return rendered_history_elements, stored_chat_data, 0

# --- Clientside Callback for Auto-Scrolling ---
clientside_callback(
    """
    function(data) {
        // This function is triggered by the 'scroll-trigger' data change
        // It needs to be robust, so it only attempts to scroll if the marker exists.
        const marker = document.getElementById('chat-end-marker');
        if (marker) {
            marker.scrollIntoView({ behavior: 'smooth' }); // 'smooth' for animated scroll
        }
        return window.dash_clientside.no_update; // Don't update any Dash output
    }
    """,
    Output('scroll-trigger', 'data', allow_duplicate=True), # Dummy output to trigger the clientside callback
    Input('scroll-trigger', 'data'),   # Input is the data from our Python callback
    prevent_initial_call=True, # Prevent scrolling on initial page load from this callback
)