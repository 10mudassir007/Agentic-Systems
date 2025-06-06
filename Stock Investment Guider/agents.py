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
    
    Now provide the list.
    """

    # desc_1 = """
    # This agent takes user-specific inputs including interests, risk tolerance, and location, then generates a curated list of 10 to 20 publicly traded companies with corresponding investment ranges and ticker symbols. The output is strictly formatted for downstream processing and avoids any explanatory or descriptive text. The agent's role is purely functional: extract relevant stock data using the given structured prompt and output it in the exact required format. 
    # USE THE TOOL PROVIDED 
    
    # """

    expected_out_1 = """
    Company Name : Ticker : Investment Range
    Company 1 : Ticker 1: 300- 400
    Company 2 : Ticker 2: 500- 1000
    Company 3 : Ticker 3: 100- 200
    """

    return Agent(
        model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
        tools=[TavilyTools(max_tokens=3000,)], 
        show_tool_calls=True,
        #description=desc_1,
        instructions=search_prompt,
        expected_output=expected_out_1,
        markdown=True
    )

def create_filter_agent():
    """Create and return the filtering agent"""
    filter_inst = """
    User Budget: {budget}
    Domains List: {domains}
    Location: {user_location}
    Currency: {currency}

    USE THE TOOL PROVIDED 
    
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

    Now return the filtered list.
    """

    # filter_desc = """
    # This agent processes a list of domain investment opportunities along with a user's budget and location. It filters out entries whose minimum investment exceeds the budget and normalizes currencies based on the user's location if needed. Then, using ROI metrics, it ranks the remaining options and selects the top 5 opportunities. The output is strictly formatted, with no extra text or commentary. Its purpose is to return a concise list suitable for direct downstream consumption by other agents, APIs, or UI components.
    # USE THE TOOL PROVIDED 
    
    # """

    filter_out = """
    
    Company: "{user_specified_currency}{Allocation}"(use user location currency)
    Company: "{user_specified_currency}{Allocation}"(use user location currency)
    Company: "{user_specified_currency}{Allocation}"(use user location currency)
    Company: "{user_specified_currency}{Allocation}"(use user location currency)
    Company: "{user_specified_currency}{Allocation}"(use user location currency)
    """

    return Agent(
        model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
        tools=[TavilyTools(max_tokens=3000,)], 
        show_tool_calls=True,
        #description=filter_desc,
        instructions=filter_inst,
        expected_output=filter_out,
        markdown=True
    )

def create_final_agent():
    """Create and return the final allocation agent"""
    final_inst = """
    Domains: {filtered_domains}
    Investment Budget: {investment_budget}
    Currency: {currency}

    USE THE TOOL PROVIDED 
    
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

    # final_desc = """
    # This agent evaluates a list of filtered investment domains and a given investment budget. It analyzes the optimal way to distribute the user's capital across the stocks—whether through equal allocation, concentrated bets, or a hybrid approach—based on estimated ROI and risk balance. The output consists of a simple strategy label, a breakdown of allocation amounts per company, and a rough projected return range. The agent's role is to produce a clear, structured recommendation strictly following a predefined format, without commentary or deviation. Output only the final answer no description, no steps etc
    # USE THE TOOL PROVIDED 
    
    # """

    # final_out = """

    # """

    return Agent(
        model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
        tools=[TavilyTools(max_tokens=3000,)], 
        show_tool_calls=True,
        #description=final_desc,
        instructions=final_inst,
        #expected_output=final_out,
        markdown=True
    )

def run_agent_pipeline(user_interests: str, budget: float, risk_tolerance: str):
    """
    Run the complete agent pipeline and return results
    """
    location = get_user_location()
    
    # Step 1: Get initial stock recommendations
    initial_agent = create_initial_agent()
    initial_agent_res = initial_agent.run(str({
        "user_interests": user_interests,
        "risk_tolerance": risk_tolerance,
        "location": location,
        "currency":currency
    }))
    domains = initial_agent_res.content
    
    # Step 2: Filter stocks based on budget
    filter_agent = create_filter_agent()
    filter_agent_res = filter_agent.run(str({
        "budget": budget,
        "domains": domains,
        "user_location": location,
        "currency":currency
    }))
    filtered_domains = filter_agent_res.content.split('\n\n')[-1]
    
    # Step 3: Generate final allocation strategy
    final_agent = create_final_agent()
    final_agent_res = final_agent.run(str({
        "filtered_domains": filtered_domains,
        "investment_budget": budget,
        "currency":currency
    }))
    
    return {
        "initial_stocks": domains,
        "filtered_stocks": filtered_domains,
        "final_strategy": final_agent_res.content,
        "location": location
    }