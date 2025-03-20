# from langchain_community.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
import os

os.environ["GOOGLE_API_KEY"] = ""
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
user_query = "Generate a Python script to plot a bar chart showing year vs number of job losses using Matplotlib and save it as 'output/graph.png'"
messages = [
    SystemMessage(content="You are an expert Python developer. Generate correct and executable Python code."),
    HumanMessage(content=user_query),
] #LLM prompt

response = llm(messages)
generated_code = response.content

file_path = "src/generated_graph.py"
with open(file_path, "w") as f:
    f.write(generated_code)

print(f"Python code generated and saved to {file_path}")