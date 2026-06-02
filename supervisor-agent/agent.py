import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.agents import create_agent
    from langchain.tools import tool
    from tavily import TavilyClient
    from dotenv import load_dotenv
    import os

    return (
        ChatGoogleGenerativeAI,
        TavilyClient,
        create_agent,
        load_dotenv,
        os,
        tool,
    )


@app.cell
def _(load_dotenv):
    load_dotenv(dotenv_path=r"F:\Files\Portfolio\.env")
    return


@app.cell
def _(os):
    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    return GOOGLE_API_KEY, TAVILY_API_KEY


@app.cell
def _(TAVILY_API_KEY, TavilyClient, tool):
    @tool
    def tavily_search(query: str):
        """Tool to search the internet"""
        client = TavilyClient(api_key=TAVILY_API_KEY)
        res = client.search(query,search_depth='advanced',max_results=5)
        return res

    return (tavily_search,)


@app.cell
def _(tavily_search):
    tools = [tavily_search]
    return (tools,)


@app.cell
def _():
    research_prompt = "You are a simple QNA assistant, your task is to use the tool provided to answer the user queries"
    return (research_prompt,)


@app.cell
def _(ChatGoogleGenerativeAI, GOOGLE_API_KEY):
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite",api_key=GOOGLE_API_KEY,max_tokens=1024)
    fallback_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=GOOGLE_API_KEY,max_tokens=1024)
    return (llm,)


@app.cell
def _(create_agent, llm, research_prompt, tools):
    research_agent = create_agent(model=llm, tools=tools, system_prompt=research_prompt)
    return (research_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## <b> It begins here <b>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Math Agent
    """)
    return


@app.cell
def _():
    math_prompt = "You are an expert mathematician, your task is to answer the user queries related to maths using the tools provided"
    return (math_prompt,)


@app.cell
def _(tool):
    @tool
    def add(x:int,y:int):
        "add two numbers"

        return x +y

    @tool
    def subtract(x:int,y:int):
        "subtract two numbers"

        return x -y

    @tool
    def multiply(x:int,y:int):
        "multiply two numbers"

        return x * y

    @tool
    def divide(x:int,y:int):
        "divide two numbers"
        try:
            return x / y
        except:
            return "Error in division"

    math_tools = [add,subtract, multiply, divide]
    return (math_tools,)


@app.cell
def _(create_agent, llm, math_prompt, math_tools):
    math_agent = create_agent(model=llm, tools=math_tools, system_prompt=math_prompt)
    return (math_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    writer agaent
    """)
    return


@app.cell
def _():
    writer_prompt="""You are a professional Writer Agent.
    Your only job is to take raw or rough content and transform it into:
    - Clear, well-structured text
    - Correct grammar and flow
    - Appropriate tone for the context (formal/casual based on query)
    - Concise and readable output

    Do not add information that wasn't in the original. Just polish and structure."""
    return (writer_prompt,)


@app.cell
def _(create_agent, llm, writer_prompt):
    writer_agent = create_agent(model=llm, system_prompt=writer_prompt)
    return (writer_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Orchestration
    """)
    return


@app.cell
def _(math_agent, research_agent, tool, writer_agent):
    @tool
    def search_web(query: str) -> str:
        "Search the web Internet Agent"
        response = research_agent.invoke({"messages":query})
        return response['messages'][-1].content

    @tool
    def perform_calculations(query: str) -> str:
        "Perform any DMAS operation using this agent"
        response = math_agent.invoke({"messages":query})
        return response['messages'][-1].content

    @tool
    def write_content(query: str) -> str:
        "Write any content about any topic e.g. essays, paragraphs etc"
        response = writer_agent.invoke({"messages":query})
        return response['messages'][-1].content


    supervisor_tools = [search_web, perform_calculations, write_content]
    return (supervisor_tools,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Supervisor
    """)
    return


@app.cell
def _():
    supervisor_prompt = "You are a supervisor agent for sub agents, your task is to read the query, classify the intent and route the query to the correct subagent"
    return (supervisor_prompt,)


@app.cell
def _(create_agent, llm, supervisor_prompt, supervisor_tools):
    supervisor_agent = create_agent(model=llm,tools=supervisor_tools, system_prompt=supervisor_prompt)
    return (supervisor_agent,)


@app.cell
def _(supervisor_agent):
    query = "What is the situation is Pakistan today?"

    for step in supervisor_agent.stream(
        {"messages": [{"role": "user", "content": query}]}
    ):
        for update in step.values():
            for message in update.get("messages", []):
                message.pretty_print()
    return


if __name__ == "__main__":
    app.run()
