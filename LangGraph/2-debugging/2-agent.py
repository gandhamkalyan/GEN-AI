from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import START,END
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import tools_condition
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_TRACING'] = "true"
os.environ['LANGSMITH_PROJECT'] = 'TestProject'

llm = init_chat_model(
    model="groq:openai/gpt-oss-120b"
)

class State(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

def make_tool_graph():
    
    @tool
    def add(a:float,b:float):
        """Add two Number"""
        return a+b

    tools=[add]

    llm_with_tool = llm.bind_tools([add])

    def call_llm_models(state:State):
        return {"messages":[llm_with_tool.invoke(state['messages'])]}


    builder = StateGraph(State)
    builder.add_node("tool_calling_llm",call_llm_models)
    builder.add_node("tools",ToolNode(tools))

    builder.add_edge(START,'tool_calling_llm')
    builder.add_conditional_edges('tool_calling_llm',tools_condition)
    builder.add_edge("tools","tool_calling_llm")

    graph = builder.compile()
    return graph

tool_agent = make_tool_graph()