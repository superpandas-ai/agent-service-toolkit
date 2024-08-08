import math
import numexpr
import re
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults, ArxivQueryRun

web_search = DuckDuckGoSearchResults()

# Kinda busted since it doesn't return links
arxiv_search = ArxivQueryRun()

@tool
def calculator(expression: str) -> str:
    """Calculates a math expression using numexpr.
    
    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )
