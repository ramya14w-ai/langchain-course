from dotenv import load_dotenv
# MessagesState, which is a predefined state class used in LangGraph
# to store and manage a list of chat messages (user, assistant, system, etc.)
from langgraph.graph import MessagesState

# ToolNode, a helper node that allows an LLM to call tools
# within a LangGraph workflow (e.g., search, calculator, APIs)
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv()

# A system message that tells the AI how it should behave
# This message sets the rules or personality of the assistant
SYSTEM_MESSAGE="""
You are a helpful assistant that can use tools to answer questions.
"""

# This function runs one step of the agent's reasoning
# It takes the current conversation state as input
def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """

    # Call the language model (llm)
    # We send:
    # 1. The system message (instructions for the AI)
    # 2. All previous messages stored in the state
    response = llm.invoke([{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]])
    # Return the updated state
    # The new AI response is added to the messages list
    return {"messages": [response]}

# Create a ToolNode
# This allows the agent to use the tools (like search, Multiplier, etc.)
tool_node = ToolNode(tools)
