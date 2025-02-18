import geocoder
import os
import time
import datetime
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.tools import TavilySearchResults
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
import streamlit as st
load_dotenv()

html_content = '''
<div style="display: flex; align-items: center;">
  <h1>Personalized Travel Planner</h1>
  <a href="https://groq.com" target="_blank" rel="noopener noreferrer" style="margin-left: 10px;">
    <img
      src="https://groq.com/wp-content/uploads/2024/03/PBG-mark1-color.svg"
      alt="Powered by Groq for fast inference."
      style="width: 120px; height: auto;"
    />
  </a>
</div>
'''
st.markdown(html_content,unsafe_allow_html=True)

query = st.text_input("Query",label_visibility='collapsed',placeholder="Plan a 3-day trip to paris")
b = st.button("Enter")
location = geocoder.ip('me')
location = location.city + ", " + location.country
today = datetime.date.today()

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")

tool = TavilySearchResults(max_results=3)

tools = [tool]

base_prompt = hub.pull("hwchase17/react")

def flight_agent(query: str):
    flight_prompt = base_prompt
    flight_prompt.template = """You are a flight guide assistant, Your task is to extract the location from the user query where the user want to go.
    Then you have to provide the flights to the location user wants to go from the closest airport from the user's location. Here is user's location: {user_location}
    If the budget is provided then provide the flights in the 25% of the budget because there are other expenses too, if budget is not provided then provide the flights in any range.
    You also have to show the prices, dates(date should be in future timespan of 1 month at most,today's date is {todays_date}) and class of flights too. Do not show any additional information except flights, nothing
    
    Output in the following format:
    Flight 1 | Price of Flight 1 | Date of Flight 1 | Class of Flight 1 | Name of Airline 1
    Flight 2 | Price of Flight 2 | Date of Flight 2 | Class of Flight 2 | Name of Airline 2
    Flight 3 | Price of Flight 3 | Date of Flight 3 | Class of Flight 3 | Name of Airline 3
    Flight 4 | Price of Flight 3 | Date of Flight 4 | Class of Flight 4 | Name of Airline 4
    Flight 5 | Price of Flight 3 | Date of Flight 5 | Class of Flight 5 | Name of Airline 5
    Show only 3-5 flights at most and only the list of flights
    You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can run only once)
    Thought: I now know the final answer
    Action: If you think you know the final answer, then generate the answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""

    llm_flight = ChatGroq(model="qwen-2.5-32b")

    agent_flight = create_react_agent(llm_flight, tools, flight_prompt)

    agent_flight_ex = AgentExecutor(agent=agent_flight, tools=tools, verbose=True,handle_parsing_errors=True)

    flights = agent_flight_ex.invoke({"input":query,"user_location":location,'todays_date':today})

    return flights

def hotel_agent(query : str):
    hotel_prompt = base_prompt
    hotel_prompt.template = """You are a hotel guide assistant, Your task is to extract the location from the user query where the user want to go.
    If the budget is provided then provide the hotels in 25% of the budget if budget is not provided then provide the hotels in any range.
    You also have to the exact prices of hotels too.
    Output in the following format:
    Hotel 1 | Name of Hotel 1 | Price of Hotel 1
    Hotel 2 | Name of Hotel 2 | Price of Hotel 2
    Hotel 3 | Name of Hotel 3 | Price of Hotel 3
    Show only 7-8 hotels at most and only the list
    You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat only 1 time)
    Thought: I now know the final answer
    Action: If you think you know the final answer, then generate the answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""

    llm_hotel = ChatGroq(model="qwen-2.5-32b")
    agent_hotel = create_react_agent(llm_hotel, tools, hotel_prompt)

    agent_hotel_ex = AgentExecutor(agent=agent_hotel, tools=tools, verbose=True,handle_parsing_errors=True)

    hotels = agent_hotel_ex.invoke({"input":query})

    return hotels

def tour_agent(query: str):
    tourist_prompt = base_prompt
    tourist_prompt.template = """You are a tour guide assistant, Your task is to extract the location from the user query where the user wants to go and
    based on the location provide the 10 most popular tourist attractions in that place/city. Just provide the names of the tourist attractions nothing else

    Output should be in a list
    Tourist Attraction 1
    Tourist Attraction 2
    Tourist Attraction 3

    ...
    You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat only 1 time)
    Thought: I now know the final answer
    Action: If you think you know the final answer, then generate the answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""

    llm_tourist = ChatGroq(model="qwen-2.5-32b")
    agent_tourist = create_react_agent(llm_tourist, tools, tourist_prompt)

    agent_tourist_ex = AgentExecutor(agent=agent_tourist, tools=tools, verbose=True,handle_parsing_errors=True)

    tour = agent_tourist_ex.invoke({"input":query})

    return tour

def final_agent(query: str):
    template = """You are a tour guide assistant, Your task is to extract the location from the user query where the user wants to go. Your task is to create a 
    detailed tour plan, with the data from previous agents you have flights to the location, you have hotels for that location and tourism places. If the budget is provided in
    the query then planned tour should be in budget. You just have to plan the tour with just the information provided no need to add anything else such as restaurant or transport and
    prices for the tourist spots should not be.

    Here is the data required:
    Query: {query}
    Flights : {flights}

    Hotels : {hotels}

    Tourism Places : {tourism_places}

    Output should be in string markdown


    """
    final_prompt = PromptTemplate.from_template(template)

    final_agent = final_prompt | ChatGroq(model="deepseek-r1-distill-llama-70b")

    flights = flight_agent(query)
    hotels = hotel_agent(query)
    tours = tour_agent(query)

    plan = final_agent.invoke({'query':query,'flights':flights, "hotels":hotels, "tourism_places":tours})

    return plan.content.split("</think>")[-1]

def generate(string : str):
    for i in string:
        yield i
        time.sleep(0.01)

if b:
    final_plan = final_agent(query)
    st.write_stream(generate(final_plan))
    