import os
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.tavily import TavilyTools
from utils import get_user_location, get_currency
# Set up environment
os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

user_location = get_user_location()
currency = get_currency(user_location)

def create_initial_agent():
    """Create and return the initial stock search agent"""
    search_prompt = """
    User Interests: {user_interests}
    Risk Tolerance: {risk_tolerance}
    Location: {location}
    Currency: {currency}

    USE THE TOOL PROVIDED 
    
    USE USER SPECIFIED LOCATION'S CURRENCY
    
    Based on the above data, use the provided tool to search for 10 to 20 stocks suitable for investment.

    USE THE TOOL PROVIDED 
    
    USE USER SPECIFIED LOCATION'S CURRENCY
    
    Output format (strictly follow this, no deviations):

    Company Name : Investment Range(in specified location currency) : Ticker Symbol

    Requirements:
    - Output ONLY the stock info in the specified format, nothing else.
    - Company names must be correctly and fully spelled.
    - Investment ranges must be diverse, starting as low as 10 and going up to 100,000.
    - Do NOT include any descriptive text, explanations, or additional information.
    - Each stock should be on its own line.
    - Try to include as much stocks as possible from the location specified above
    
    You must answer in exactly one step. 
    Do not generate multiple Thought/Action loops. 
    Now provide the list.
    """

    expected_out_1 = """
    Company Name : Ticker : Investment Range
    Company 1 : Ticker 1: 300- 400
    Company 2 : Ticker 2: 500- 1000
    Company 3 : Ticker 3: 100- 200
    """

    return Agent(
        model=Groq(id="openai/gpt-oss-20b"),
        tools=[TavilyTools(max_tokens=1000,search_depth="basic")], 
        instructions=search_prompt,
        expected_output=expected_out_1,
        markdown=True,
        tool_call_limit=1
    )

def create_filter_agent():
    """Create and return the filtering agent"""
    filter_inst = """
    User Budget: {budget}
    Domains List: {domains}
    Location: {user_location}
    Currency: {currency}

    USE THE TOOL PROVIDED ONLY 
    
    From the above domains, perform the following:

    - Exclude any domain where the **minimum investment amount** exceeds the **user's budget**.
    - If the currency is not specified, infer and apply the **currency based on the user's location**.
    - From the remaining entries, identify the **top 5** domains with the **highest return on investment (ROI)**.
    - Use the appropriate tool to determine ROI.
    - Output the **final top 5 filtered domains only**, in the exact format below.

    USE THE TOOL PROVIDED 
    
    USE USER SPECIFIED LOCATION'S CURRENCY
    
    FORMAT (strict):
    Company Name : Ticker : Investment Range

    RULES:
    - Output only the final list.
    - No explanations, no commentary, and no extra text.
    - Use fully spelled-out company names.
    - Each entry must be on its own line, exactly matching the format.
    - Convert the currency to the user's location currency
    - Only Output the final answer no description, no steps

    You must answer in exactly one step. 
    Do not generate multiple Thought/Action loops. 

    Now return the filtered list.
    """

    filter_out = """
    
    Company: "{user_specified_currency}{Allocation}"(use user location currency)
    Company: "{user_specified_currency}{Allocation}"(use user location currency)
    Company: "{user_specified_currency}{Allocation}"(use user location currency)
    Company: "{user_specified_currency}{Allocation}"(use user location currency)
    Company: "{user_specified_currency}{Allocation}"(use user location currency)
    """

    return Agent(
        model=Groq(id="openai/gpt-oss-20b"),
        tools=[TavilyTools(max_tokens=1000,search_depth="basic")], 
        instructions=filter_inst,
        expected_output=filter_out,
        markdown=True,
        tool_call_limit=1
    )

def create_final_agent():
    """Create and return the final allocation agent"""
    final_inst = """
    Domains: {filtered_domains}
    Investment Budget: {investment_budget}
    Currency: {currency}

    USE THE TOOL PROVIDED,
    
    USE USER SPECIFIED LOCATION'S CURRENCY
    
    TASK:
    Analyze the provided domains and the user's total investment budget. Determine the **most strategic way** to allocate the investment across the listed stocks. Consider diversification, risk-adjusted return potential, and practical allocation boundaries.

    INSTRUCTIONS:
    - Recommend whether the user should:
      - Distribute funds evenly across all stocks,
      - Concentrate the investment in one or a few high-potential stocks,
      - Use a mixed strategy (e.g., core-satellite or tiered allocation).
    - Clearly list the recommended allocation amounts for each company.
    - Estimate a **realistic total return range** based on the selected strategy and assumed ROI performance.
    - Use approximate figures; the user expects a rough but actionable plan.

    FORMAT:
    Strategy: [brief explanation]
    Allocations:
    Company A : X
    Company B : Y
    ...
    Estimated Total Return: LOW - HIGH

    RULES:
    - Output only the result in the format above.
    - No extra commentary, disclaimers, or metadata.
    - Use the full company names and keep the formatting clean and readable.
    
    You must answer in exactly one step. 
    Do not generate multiple Thought/Action loops. 
    
    - Output only the final answer no description, no steps etc
    
        
    The currency should be as provided in the domains
    Allocations:
    Company: "{user_specified_currency}{Allocation}"
    Company: "{user_specified_currency}{Allocation}"
    Company: "{user_specified_currency}{Allocation}"
    Company: "{user_specified_currency}{Allocation}"
    Company: "{user_specified_currency}{Allocation}"
    
    or 
    
    Allocations:
    Company X : All amount

    Estimated Total Return: "{user_specified_currency}{Total Returns}"
    """

    return Agent(
        model=Groq(id="openai/gpt-oss-20b"),
        tools=[TavilyTools(max_tokens=1000,search_depth="basic")], 
        instructions=final_inst,
        markdown=True,
        tool_call_limit=1
    )

def run_agent_pipeline(user_interests: str, budget: float, risk_tolerance: str):
    """
    Run the complete agent pipeline and return results
    """
    print(user_interests, budget, risk_tolerance)
    location = get_user_location()
    
    initial_agent = create_initial_agent()
    print("[Pipeline] Running initial agent...")
    initial_agent_res = initial_agent.run(str({
        "user_interests": user_interests,
        "risk_tolerance": risk_tolerance,
        "location": location,
        "currency":currency
    }))
    print("[Pipeline] Initial agent output:", initial_agent_res.content)
    domains = initial_agent_res.content
    
    filter_agent = create_filter_agent()
    print("[Pipeline] Running filter agent...")
    filter_agent_res = filter_agent.run(str({
        "budget": budget,
        "domains": domains,
        "user_location": location,
        "currency":currency
    }))
    filtered_domains = filter_agent_res.content.split('\n\n')[-1]
    
    final_agent = create_final_agent()
    final_agent_res = final_agent.run(str({
        "filtered_domains": filtered_domains,
        "investment_budget": budget,
        "currency":currency
    }))
    print("[Pipeline] Final agent output:", final_agent_res.content)
    return {
        "initial_stocks": domains,
        "filtered_stocks": filtered_domains,
        "final_strategy": final_agent_res.content,
        "location": location
    }