from dotenv import load_dotenv

# Imports HumanMessage, which represents a message sent by the user
# (used to add user input into the conversation)
from langchain_core.messages import HumanMessage

# - MessagesState → stores the conversation history (all messages)
# - StateGraph → used to build a workflow/graph of steps (nodes)
# - END → marks where the graph should stop running
from langgraph.graph import MessagesState, StateGraph,END

# Imports custom nodes from your local 'nodes' file
# - run_agent_reasoning → function where the AI thinks and responds
# - tool_node → node that allows the AI to use tools
from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON="agent_reason"
ACT= "act"
LAST = -1

# Function that decides whether the graph should continue or stop
def should_continue(state: MessagesState) -> str:
    # Check the last message from the AI
    # If it did NOT request any tool calls, stop the workflow
    if not state["messages"][LAST].tool_calls:
        return END
    # If the AI wants to use a tool, go to the ACT node
    return ACT

# Create a new state graph that uses MessagesState
# This graph controls how the agent flows between steps
flow = StateGraph(MessagesState)

# Add the agent reasoning node to the graph
# This is where the AI thinks and generates a response
flow.add_node(AGENT_REASON, run_agent_reasoning)
# Set the starting point of the graph
flow.set_entry_point(AGENT_REASON)
# Add the action node
# This node is responsible for executing tool calls
flow.add_node(ACT, tool_node)

# Add conditional edges from the agent reasoning node
# Based on should_continue():
# END → stop the graph
# ACT → run tools
flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END:END,
    ACT:ACT})

# After tools are executed, go back to the agent reasoning step
# This allows the AI to see tool results and think again
flow.add_edge(ACT, AGENT_REASON)

# Compile the graph into a runnable app
app = flow.compile()
# Generate a visual diagram of the graph
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling")
    res = app.invoke({"messages": [HumanMessage(content="What is the temperature in Tokyo? List it and then triple it")]})
    print(res["messages"][LAST].content)