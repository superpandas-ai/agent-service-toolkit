from typing import Annotated
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from typing import Literal, Dict
from typing_extensions import TypedDict
# from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import Tool
from langgraph.graph import MessagesState
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
        f"This conversation was flagged for unsafe content: "
        f"{', '.join(safety.unsafe_categories)}"
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
        return {"success": True, "output": result}
    except BaseException as e:
        return {"success": False, "error": repr(e)}


# LLM
groq = ChatGroq(model='llama-3.3-70b-versatile')

class TaskClassifier(TypedDict):
    task_type: Literal["pandas", "matplotlib", "sklearn"]

def classify_task(state: MessagesState) -> Command[Literal["pandas_agent", "matplotlib_agent", "sklearn_agent"]]:
    
    system_prompt = (
        "You are a task classifier. Given the user's request, classify it into one of the following categories: "
        "- 'pandas': for data analysis, filtering, and transformations. "
        "- 'matplotlib': for data visualizations or plotting. "
        "- 'sklearn': for machine learning model training, evaluation, or prediction."
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = groq.with_structured_output(TaskClassifier).invoke(messages)
    task_type = response["task_type"]

    if task_type == "pandas":
        return Command(goto="pandas_agent")
    elif task_type == "matplotlib":
        return Command(goto="matplotlib_agent")
    elif task_type == "sklearn":
        return Command(goto="sklearn_agent")
    return Command(goto=END)


def make_react_node(llm: BaseChatModel,
                    state_modifier: str,
                    node_name: str,
                    tools: list[Tool] = [python_repl_tool]) -> str:

    agent = create_react_agent(
        llm, tools=tools, state_modifier=state_modifier
    )

    def node(state: MessagesState) -> Command:
        result = agent.invoke(state)
        response_message = result["messages"][-1].content

        # Properly handle success and failure
        if "success" in response_message and "error" not in response_message:
            return Command(update={"messages": [HumanMessage(content=response_message)]}, goto=END)

        return Command(update={"messages": [HumanMessage(content="An error occurred during processing")]}, goto=END)

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
                              state_modifier=(
        "You are a data analysis expert specializing in the Python library Pandas. "
        "Your task is to write efficient, clean, and optimized code for data manipulation, aggregation, filtering, and analysis. "
        "Whenever you generate code, ensure that: "
        "- The data is handled using appropriate Pandas functions. "
        "- Edge cases like missing or duplicate data are properly handled. "
        "- The output is well-structured and easy to interpret."
    ),
                              node_name='pandas_agent')
matplotlib_node = make_react_node(llm=groq,
                                  state_modifier=(
        "You are a data visualization expert specializing in the Python library Matplotlib. "
        "Your task is to generate clear, informative, and visually appealing plots based on user-provided data or requirements. "
        "Whenever you generate plots, ensure that: "
        "- The appropriate plot type (e.g., bar, scatter, line) is selected based on the data. "
        "- Titles, labels, and legends are correctly added to enhance readability. "
        "- Customizations (e.g., colors, gridlines) are applied when necessary."
    ),
                                  node_name='matplotlib_agent')
sklearn_node = make_react_node(llm=groq,
                               state_modifier=(
        "You are a machine learning expert specializing in the Python library scikit-learn. "
        "Your task is to generate code for model training, evaluation, and predictions using appropriate algorithms and preprocessing steps. "
        "Whenever you generate machine learning code, ensure that: "
        "- The dataset is preprocessed correctly (e.g., missing data handling, feature scaling). "
        "- The model selection is based on the given task (classification, regression, etc.). "
        "- The output includes model evaluation metrics (e.g., accuracy, F1-score)."
    ),
                               node_name='sklearn_agent')

def supervisor_node(state: MessagesState) -> Command[Literal["FINISH"]]:
    return Command(goto=END)

agent = StateGraph(AgentState)
agent.add_node("guard_input", llama_guard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.add_node("classify_task", classify_task)
agent.add_node("pandas_agent", pandas_node)
agent.add_node("matplotlib_agent", matplotlib_node)
agent.add_node("sklearn_agent", sklearn_node)

agent.set_entry_point("guard_input")

# Edges for safety and task routing
agent.add_conditional_edges("guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "classify_task"})
agent.add_edge("pandas_agent", END)
agent.add_edge("matplotlib_agent", END)
agent.add_edge("sklearn_agent", END)
agent.add_edge("block_unsafe_content", END)

spd_agent = agent.compile(checkpointer=MemorySaver())