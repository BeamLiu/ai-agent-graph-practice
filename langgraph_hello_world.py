import io
import os

from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from PIL import Image as PILImage

MODEL_ID = "gemini-2.0-flash-exp"
_ = load_dotenv()

tool = TavilySearchResults(max_results=4)
print(type(tool))
print(tool.name)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    def __init__(self, model, tools, system=''):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node('llm', self.call_llm)
        graph.add_node('action', self.execute_action)
        graph.add_conditional_edges(
            'llm',
            self.action_exists,
            {True: 'action', False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point('llm')
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_llm(self, state: AgentState):
        msgs = state['messages']
        if self.system:
            msgs = [SystemMessage(content=self.system)] + msgs
        message = self.model.invoke(msgs)
        return {'messages': [message]}

    def execute_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f'calling {t}')
            if not t['name'] in self.tools:
                print(f'cannot find tool {t.name}')
                return 'bad tool name, please retry'
            else:
                res = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(res)))
        print("Back to the model!")
        return {'messages': results}

    def action_exists(self, state: AgentState):
        res = state['messages'][-1]
        return len(res.tool_calls) > 0


prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

gemini_model = ChatOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GEMINI_API_KEY"),
    model=MODEL_ID
)
abot = Agent(gemini_model, [tool], system=prompt)

messages = [HumanMessage(content="What is the weather in sf?")]
result = abot.graph.invoke({"messages": messages})

print(result)
print(result['messages'][-1].content)
messages = [HumanMessage(content="What is the weather in SF and LA?")]
result = abot.graph.invoke({"messages": messages})
print(result['messages'][-1].content)

query = "Who won the super bowl in 2024? In what state is the winning team headquarters located? \
What is the GDP of that state? Answer each question."
messages = [HumanMessage(content=query)]

abot = Agent(gemini_model, [tool], system=prompt)
result = abot.graph.invoke({"messages": messages})
print(result['messages'][-1].content)
