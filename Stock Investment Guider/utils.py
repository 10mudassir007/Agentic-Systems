import geocoder
from groq import Groq
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()

os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

def get_user_location():
    """Get user location based on IP"""
    g = geocoder.ip('me')
    return g.country


def get_currency(location):
    client = Groq()
    completion = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[
            {
                "role": "user",
                "content": f"What currency is used in {location}? Just respond with the currency name only."
            }
        ],
        temperature=0,
        max_completion_tokens=50,
        top_p=1,
        stream=True,
        stop=None,
        
    )

    response = ""
    for chunk in completion:
        response += chunk.choices[0].delta.content or ""

    return response

print(get_user_location())
print(get_currency("PK"))
