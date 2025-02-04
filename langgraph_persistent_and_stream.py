import asyncio
import os
import sqlite3

import aiosqlite
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

MODEL_ID = "gemini-2.0-flash-exp"
_ = load_dotenv()

tool = TavilySearchResults(max_results=2)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


memory = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        msgs = state['messages']
        if self.system:
            msgs = [SystemMessage(content=self.system)] + msgs
        message = self.model.invoke(msgs)
        if not message.content:
            message.content = 'function call'
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}


prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that directly!
"""
gemini_model = ChatOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GEMINI_API_KEY"),
    model=MODEL_ID
)
abot = Agent(gemini_model, [tool], system=prompt, checkpointer=memory)

messages = [HumanMessage(content="What is the weather in sf?")]
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({'messages': messages}, config=thread):
    for v in event.values():
        print(v['messages'])

messages = [HumanMessage(content="What about in la?")]
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)

messages = [HumanMessage(content="Which one is warmer?")]
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)

messages = [HumanMessage(content="Which one is warmer?")]
thread = {"configurable": {"thread_id": "2"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)


# streaming tokens
# Define the async function
async def run_streaming():
    # Initialize memory and agent for async
    memory = AsyncSqliteSaver(aiosqlite.connect(":memory:", check_same_thread=False))
    abot = Agent(gemini_model, [tool], system=prompt, checkpointer=memory)

    messages = [HumanMessage(content="What is the weather in SF?")]
    thread = {"configurable": {"thread_id": "4"}}

    # Now use async for inside an async function
    async for event in abot.graph.astream_events({"messages": messages}, thread, version="v2"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # Empty content in the context of OpenAI means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                print(content, end="|")


# Run the async function with asyncio
asyncio.run(run_streaming())
