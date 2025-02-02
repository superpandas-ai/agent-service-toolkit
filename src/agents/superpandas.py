from typing import Annotated
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

from typing import Literal, Dict
from typing_extensions import TypedDict

# from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import Tool

from langgraph.graph import MessagesState, END
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.managed import RemainingSteps
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment

# Model Safety Guard


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: LlamaGuardOutput
    # remaining_steps: RemainingSteps
    next: str


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {
            ', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    llama_guard = LlamaGuard()
    safety_output = llama_guard.invoke("User", state["messages"])
    return {"safety": safety_output}


def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}

# Python Tool


repl = PythonREPL()


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code and do math. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"

# LLM


groq = ChatGroq(model='llama-3.3-70b-versatile')

# Nodes and Agents

members = ["pandas_agent", "matplotlib_agent", "sklearn_agent"]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]


def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]  # type: ignore

    # type: ignore
    def supervisor_node(state: MessagesState) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto)

    return supervisor_node


def make_react_node(llm: BaseChatModel,
                    state_modifier: str,
                    node_name: str,
                    tools: list[Tool] = [python_repl_tool],) -> str:

    agent = create_react_agent(
        llm, tools=tools, state_modifier=state_modifier
    )

    def node(state: MessagesState) -> Command[Literal["supervisor"]]:
        result = agent.invoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=result["messages"][-1].content, name=node_name)
                ]
            },
            goto="supervisor",
        )
    return node

# Check for unsafe input and block further processing if found


def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


pandas_node = make_react_node(llm=groq,
                              state_modifier="You are a data analyst expert in python library pandas",
                              node_name='pandas_agent')
matplotlib_node = make_react_node(llm=groq,
                                  state_modifier="You are a plotting and visualization expert in python library matplotlib",
                                  node_name='matplotlib_agent')
sklearn_node = make_react_node(llm=groq,
                               state_modifier="You are a machine learning expert in python library scikit-learn",
                               node_name='sklearn_agent')

supervisor_node = make_supervisor_node(
    groq, ["pandas_agent", "matplotlib_agent", "sklearn_agent"])

agent = StateGraph(AgentState)
agent.add_node("guard_input", llama_guard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.add_node("supervisor", supervisor_node)
agent.add_node("pandas_agent", pandas_node)
agent.add_node("matplotlib_agent", matplotlib_node)
agent.add_node("sklearn_agent", sklearn_node)
agent.set_entry_point("guard_input")

# Add conditional edges to the graph
agent.add_conditional_edges(
    "guard_input", check_safety, {
        "unsafe": "block_unsafe_content", "safe": "supervisor"}
)
# Always END after blocking unsafe content
agent.add_edge("block_unsafe_content", END)

# checkpointer=MemorySaver())
spd_agent = agent.compile(checkpointer=MemorySaver())
