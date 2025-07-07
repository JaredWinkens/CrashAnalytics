import asyncio
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult
from pydantic import BaseModel
import json

# load config settings
config_file = open("config.json", "r")
config = json.load(config_file)
API_KEY = config['general']['api_key']
GEN_MODEL = config['models']['2.5-flash']
MCP_SERVER_COMMAND = "python"
MCP_SERVER_ARGS = ["chatbot/mcp_server.py"]

gemini_client = genai.Client(api_key=API_KEY)

_mcp_session: ClientSession = None
_gemini_chat = None

system_prompt = """
You are an interactive roadway safety chatbot designed to provide users with real-time, data-driven insights on roadway safety.

Based on the user's question, determine which tool (or combination of tools) to use.
If you need to understand the structure or schema of the database use the `get_sqlite_table_schema` tool.
If a precise count, sum, average, or specific filtered data from the database is needed, use the `execute_read_sqlite_query` tool.
If a general description, summary, or narrative insight is needed, use the `search_chroma_documents` tool.
If the user asks for a map, visualization, or geographical representation of data, use the `visualize_data` tool.
If the question is just for general advice and can be answered without any tools, do so.
If the question is outside the scope of your purpose, state that.
"""

async def _initialize_mcp_and_gemini():
    print("Starting Gemini-MCP Client...")
    global _mcp_session, _gemini_chat

    if _mcp_session is not None and _gemini_chat is not None:
        return # Already initialized

    print("Initializing MCP and Gemini...")
    server_params = StdioServerParameters(
        command=MCP_SERVER_COMMAND,
        args=MCP_SERVER_ARGS
    )
    tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY", #allowed_function_names=["get_current_temperature"]
                )
            )
    _gemini_chat = gemini_client.aio.chats.create(
                model=GEN_MODEL,
                config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0,
                        tools=[_mcp_session],
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        ),
                        #tool_config=tool_config
                    )
                )
    print("Gemini chat initialized.")

async def get_gemini_response_from_mcp(user_message: str) -> dict[str, any]:
    """
    Sends a user message to Gemini, handles tool calls via MCP,
    and returns Gemini's response along with any visualization data.
    """
    global _mcp_session, _gemini_chat

    if _mcp_session is None or _gemini_chat is None:
        # Re-establish connection if not active (important for background callbacks)
        server_params = StdioServerParameters(
            command=MCP_SERVER_COMMAND,
            args=MCP_SERVER_ARGS
        )
        # Re-initialize MCP session and Gemini chat for each call in background callback
        # This ensures fresh state and avoids issues with long-lived async objects in multiprocess Dash.
        # For a high-performance app, consider a shared process or message queue for MCP.
        _mcp_session = ClientSession(
            (await asyncio.create_subprocess_exec(
                server_params.command, *server_params.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )).stdout,
            (await asyncio.create_subprocess_exec(
                server_params.command, *server_params.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )).stdin
        )
        await _mcp_session.initialize()
        print("MCP session re-established for background callback.")
        await _initialize_mcp_and_gemini() # Ensure Gemini chat is also ready

    bot_response_text = ""
    visualization_data = None
    
    try:
        # Send user query to Gemini
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as mcp_session:
                await mcp_session.initialize()
                print("MCP Session initialized for request.")
                schema = await _get_all_table_schemas(_mcp_session)
                print(schema)
                await _gemini_chat.send_message(f"Here is the schema of the SQL Database\n\n{schema}")
                response = await _gemini_chat.send_message(user_message)
            
                # Process Gemini's response
                # If Gemini decided to call a tool, the response will contain tool_calls
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.function_call:
                            tool_call = part.function_call
                            tool_name = tool_call.name
                            tool_args = tool_call.args

                            print(f"Gemini decided to call tool: {tool_name} with arguments: {tool_args}")

                            # Execute the tool based on Gemini's request
                            if tool_name == "execute_read_sqlite_query":
                                tool_result = await _mcp_session.call_tool(name=tool_name, arguments=tool_args)
                                print(f"SQL Tool Result:\n{tool_result}")
                            
                            elif tool_result == "search_chroma_documents":
                                tool_result = await _mcp_session.call_tool(name=tool_name, arguments=tool_args)
                                print(f"RAG Tool Result:\n{tool_result}")

                            elif tool_name == "visualize_data":
                                tool_result = await _mcp_session.call_tool(name=tool_name, arguments=tool_args)
                                print(f"VIS Tool Result:\n{tool_result}")
                                visualization_data = json.loads(tool_result.structuredContent.get('plot_json', {}))
                                bot_response_text = tool_result.structuredContent.get('message')
                            
                            else:
                                tool_result = f"Unknown tool requested by Gemini: {tool_name}"
                            
                            if tool_name != "visualize_map_data":
                                await _gemini_chat.send_message(str(tool_result.structuredContent))

                            follow_up_response = await _gemini_chat.send_message(f"Tool output for {tool_name}:\n{tool_result.structuredContent}")
                            if follow_up_response.candidates and follow_up_response.candidates[0].content.parts:
                                for follow_up_part in follow_up_response.candidates[0].content.parts:
                                    if follow_up_part.text:
                                        bot_response_text += "\n" + follow_up_part.text
                        elif part.text:
                            bot_response_text += part.text
                else:
                    bot_response_text = "Gemini did not provide a text response or tool call."

    except Exception as e:
        bot_response_text = f"An error occurred during interaction: {e}"
        print(f"Error in get_gemini_response_from_mcp: {e}")

    return {"text": bot_response_text, "visualization_data": visualization_data}

async def _get_all_table_schemas(session: ClientSession) -> str:
    try:
        content, _ = await session.read_resource("sqlite://schema/all")
        return content
    except Exception as e:
        return f"Error fetching all SQLite table schemas: {e}"

if __name__ == "__main__":
    async def test_client():
        # This part is just for direct testing of this script
        # In the Dash app, get_gemini_response_from_mcp will be called
        print("Running direct test of mcp_client.py...")
        await _initialize_mcp_and_gemini() # Initialize once for direct test
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'exit':
                break
            response = await get_gemini_response_from_mcp(user_input)
            print(f"Bot: {response['text']}")
            if response['visualization_data']:
                print(f"Visualization Data: {response['visualization_data']}")

    asyncio.run(test_client())
    if _mcp_session:
        # Close the session if it was opened for direct testing
        # In Dash, the session will be managed by the background callback context
        pass # The context manager will handle closing