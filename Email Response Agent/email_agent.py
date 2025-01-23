import os
from langchain_groq import ChatGroq
from crewai import Agent, Task, Process, Crew,LLM
from duckduckgo_tool import DuckDuckGo
from dotenv import load_dotenv
import time
import streamlit as st
load_dotenv()

st.set_page_config(
    page_title="AI Email Agent",
    page_icon="📧"
)
st.title("EmailEase 📫")
st.subheader("Generate Email Responses using Llama 3.3")

tool = DuckDuckGo()

llm = LLM(
    model='groq/llama-3.3-70b-versatile',
    api_key=os.getenv("GROQ_API_KEY")
)

email_categorizer = Agent(
    role="Email Categorizer Agent",
    backstory="You are a master at understanding what a customer wants when they write a email and are able to categorize it in any useful way",
    goal="""take in a email from a human that has emailed out company email address and categorize it \
            into one of the following categories: \
            price_equiry - used when someone is asking for information about pricing \
            customer_complaint - used when someone is complaining about something \
            product_enquiry - used when someone is asking for information about a product feature, benefit or service but not about pricing \\
            customer_feedback - used when someone is giving feedback about a product \
            off_topic when it doesnt relate to any other category""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    memory=True
)

researcher_agent = Agent(
    role="Info Researcher Agent",
    backstory="You are a master at understanding what information our email writer needs  to write a reply that \
            will help the customer",
    goal="""take in a email from a human that has emailed out company email address and the category \
            that the categorizer agent gave it and decide what information you need to search for for the email writer to reply to \
            the email in a thoughtful and helpful way.
            If you DONT think a search will help just reply 'NO SEARCH NEEDED'
            If you dont find any useful info just reply 'NO USEFUL RESESARCH FOUND'
            otherwise reply with the info you found that is useful for the email writer
            """,
    llm=llm,
    verbose=True,
    max_iter=5,
    allow_delegation=False,
    memory=True,
    tools=[tool]    
)

email_writer = Agent(
    role="Email Writer Agent",
    backstory="""You are a master at synthesizing a variety of information and writing a helpful email \
            that will address the customer's issues and provide them with helpful information""",
    goal="""take in a email from a human that has emailed out company email address, the category \
            that the categorizer agent gave it and the research from the research agent and \
            write a helpful email in a thoughtful and friendly way.

            If the customer email is 'off_topic' then ask them questions to get more information.
            If the customer email is 'customer_complaint' then try to assure we value them and that we are addressing their issues.
            If the customer email is 'customer_feedback' then try to assure we value them and that we are addressing their issues.
            If the customer email is 'product_enquiry' then try to give them the info the researcher provided in a succinct and friendly way.
            If the customer email is 'price_equiry' then try to give the pricing info they requested.

            You never make up information. that hasn't been provided by the researcher or in the email.
            Always sign off the emails in appropriate manner and from Sarah the Resident Manager.""",
    llm=llm,
    verbose=True,
    max_iter=5,
    allow_delegation=False,
    memory=True
)

test_email = st.text_area("Enter email")

if st.button("Enter"):
    categorizing_task = Task(
        description=f"""Conduct a comprehensive analysis of the email provided and categorize into \
                one of the following categories:
                price_equiry - used when someone is asking for information about pricing \
                customer_complaint - used when someone is complaining about something \
                product_enquiry - used when someone is asking for information about a product feature, benefit or service but not about pricing \\
                customer_feedback - used when someone is giving feedback about a product \
                off_topic when it doesnt relate to any other category \

                EMAIL CONTENT:\n\n {test_email} \n\n
                Output a single cetgory only""",
        expected_output="""A single categtory for the type of email from the types ('price_equiry', 'customer_complaint', 'product_enquiry', 'customer_feedback', 'off_topic') \
                eg:
                'price_enquiry' \
        """,
        output_file="Email_Category.txt",
        agent=email_categorizer
    )


    research_info = Task(
        description=f"""Conduct a comprehensive analysis of the email provided and the category \
                provided and search the web to find info needed to respond to the email

                EMAIL CONTENT:\n\n {test_email} \n\n
                Only provide the info needed DONT try to write the email""",
        expected_output="""A set of bullet points of useful info for the email writer \
                or clear instructions that no useful material was found.""",
        context=[categorizing_task],
        output_file="email_info.txt",
        agent=researcher_agent
    )


    draft_email = Task(
        description=f"""Conduct a comprehensive analysis of the email provided, the category provided\
                and the info provided from the research specialist to write an email. \

                Write a simple, polite and too the point email which will respond to the customer's email. \
                If useful use the info provided from the research specialist in the email. \

                If no useful info was provided from the research specialist the answer politely but don't make up info. \

                EMAIL CONTENT:\n\n {test_email} \n\n
                Output a single cetgory only""",
        expected_output="""A well crafted email for the customer that addresses their issues and concerns""",
        context=[categorizing_task,research_info],
        agent=email_writer,
        output_file='drafted_email.txt'
    )


    crew = Crew(
        agents=[email_categorizer,researcher_agent,email_writer],
        tasks=[categorizing_task,research_info,draft_email],
        verbose=1,
        process=Process.sequential,
        full_output=True,
        share_crew=False
    )

    results = crew.kickoff()

    print(crew.usage_metrics)
    def generate(string: str):
        for i in string:
            yield i + ""
            time.sleep(0.005)

    st.subheader("Generated Email Reply")
    st.write_stream(generate(results.raw))