# import matplotlib.pyplot as plt
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.schema import SystemMessage, HumanMessage
# import os
# import sys
# import importlib
# import io
# import base64
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import numpy as np
# import re

# app = FastAPI()

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class PlotRequest(BaseModel):
#     query: str

# # Set Google API key
# os.environ["GOOGLE_API_KEY"] = "AIzaSyDQcmV2pdj6Z2gSlCrWHfjojwyfOlBIv0Y"

# # Initialize LLM
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# # def extract_code_from_response(response_text: str) -> str:
    
    
# #     code_match = re.search(r'python\n(.*?)\n', response_text, re.DOTALL)
# #     if code_match:
# #         return code_match.group(1)
    
    
# #     code_match = re.search(r'(.*?)', response_text, re.DOTALL)
# #     if code_match:
# #         return code_match.group(1)
    
    
# #     return response_text

# def extract_code_from_response(response_text: str) -> str:
#     # Look for code blocks with ```python ... ``` format
#     code_block_match = re.search(r'```python\s+(.*?)\s+```', response_text, re.DOTALL)
#     if code_block_match:
#         return code_block_match.group(1).strip()
    
#     # If no code block found, look for code sections with "python" marker
#     code_section_match = re.search(r'python\s+(.*?)(?:\n\n|$)', response_text, re.DOTALL)
#     if code_section_match:
#         return code_section_match.group(1).strip()
    
#     # If still nothing found, just return the whole response as a fallback
#     # but print a warning for debugging
#     print("WARNING: Could not extract code block from response, using full response.")
#     return response_text.strip()

# def get_visualization_code(query: str):
#     """Get visualization code from Gemini"""
#     prompt = f"""
#     Generate Python code for the following visualization request: {query}

#     Requirements:
#     1. Use ONLY matplotlib and numpy (DO NOT use scipy or other libraries)
#     2. For trend lines, use numpy.polyfit instead of scipy.stats
#     3. Include all necessary imports at the top
#     4. Create sample data arrays that MUST have exactly the same length
#     5. Add proper title, labels, and grid
#     6. Make the visualization clear and professional
#     7. DO NOT include plt.show() or plt.savefig()
#     8. Use plt.figure(figsize=(15, 5)) for wide multi-panel plots

#     Example format:
#     python
#     import matplotlib.pyplot as plt
#     import numpy as np

#     # Create figure
#     plt.figure(figsize=(15, 5))

#     # For trend lines, use:
#     coefficients = np.polyfit(x, y, 1)
#     trend_line = coefficients[0] * x + coefficients[1]
    
#     """
    
#     messages = [
#         SystemMessage(content="""You are a Python data visualization expert. Generate only executable matplotlib code.
#         IMPORTANT: Use only numpy and matplotlib. For trend lines, use numpy.polyfit instead of scipy.stats."""),
#         HumanMessage(content=prompt)
#     ]
    
#     response = llm.invoke(messages)
#     code = extract_code_from_response(response.content)
#     return code.strip()


# def execute_visualization_code(code: str):
#     """Execute the generated visualization code and return base64 image"""
#     try:
#         # Clear any existing plots
#         plt.clf()
#         plt.close('all')
        
#         # Create namespace with required imports
#         namespace = {
#             'plt': plt,
#             'np': np,
#         }
        
#         # Print code for debugging
#         print("Executing code:\n", code)
        
#         # Execute the code
#         exec(code, namespace)
        
#         # Convert plot to base64
#         buf = io.BytesIO()
#         plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
#         buf.seek(0)
#         img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
#         # Clean up
#         plt.close('all')
        
#         return img_str
#     except Exception as e:
#         raise Exception(f"Error executing visualization code: {str(e)}\nCode:\n{code}")

# @app.post("/plot")
# async def generate_plot(request: PlotRequest):
#     """Generate a plot based on the user's query."""
#     try:
#         # Get visualization code from Gemini
#         code = get_visualization_code(request.query)
        
#         # Execute the code and get base64 image
#         base64_image = execute_visualization_code(code)
        
#         # Extract chart type from code (simple heuristic)
#         chart_type = "line"
#         if "plt.bar" in code:
#             chart_type = "bar"
#         elif "plt.scatter" in code:
#             chart_type = "scatter"
#         elif "plt.pie" in code:
#             chart_type = "pie"
        
#         # Return the response
#         return {
#             "image": base64_image,
#             "config": {
#                 "title": f"Visualization for: {request.query}",
#                 "chart_type": chart_type,
#                 "x_label": "X-axis",
#                 "y_label": "Y-axis"
#             }
#         }
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)





#For Chatbot purpose :
# import matplotlib.pyplot as plt
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.schema import SystemMessage, HumanMessage, AIMessage
# import os
# import io
# import base64
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import numpy as np
# import re
# import uuid
# from typing import List, Dict, Optional

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class ChatMessage(BaseModel):
#     role: str  
#     content: str
#     image: Optional[str] = None

# class ChatSession(BaseModel):
#     id: str
#     messages: List[ChatMessage]
#     current_graph_code: Optional[str] = None
#     current_graph_data: Optional[Dict] = None

# class ChatRequest(BaseModel):
#     session_id: Optional[str] = None
#     message: str

# class NewSessionRequest(BaseModel):
#     name: Optional[str] = None

# chat_sessions: Dict[str, ChatSession] = {} #chat sessions stored in dict format

# os.environ["GOOGLE_API_KEY"] = "AIzaSyDQcmV2pdj6Z2gSlCrWHfjojwyfOlBIv0Y"
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# def extract_code_from_response(response_text: str) -> str:
#     # Look for code blocks with ```python ... ``` format
#     code_block_match = re.search(r'```python\s+(.*?)\s+```', response_text, re.DOTALL)
#     if code_block_match:
#         return code_block_match.group(1).strip()
    
#     # If no code block found, look for code sections with "python" marker
#     code_section_match = re.search(r'python\s+(.*?)(?:\n\n|$)', response_text, re.DOTALL)
#     if code_section_match:
#         return code_section_match.group(1).strip()
    
#     # If still nothing found, just return the whole response as a fallback
#     print("WARNING: Could not extract code block from response, using full response.")
#     return response_text.strip()

# def get_visualization_code(query: str):
#     """Get visualization code from Gemini"""
#     prompt = f"""
#     Generate Python code for the following visualization request: {query}

#     Requirements:
#     1. Use ONLY matplotlib and numpy (DO NOT use scipy or other libraries)
#     2. For trend lines, use numpy.polyfit instead of scipy.stats
#     3. Include all necessary imports at the top
#     4. Create sample data arrays that MUST have exactly the same length
#     5. Add proper title, labels, and grid
#     6. Make the visualization clear and professional
#     7. DO NOT include plt.show() or plt.savefig()
#     8. Use plt.figure(figsize=(15, 5)) for wide multi-panel plots
#     9. SAVE THE ACTUAL DATA VALUES you generate in comments at the bottom of the code for later reference
#     """
    
#     messages = [
#         SystemMessage(content="""You are a Python data visualization expert. Generate only executable matplotlib code.
#         IMPORTANT: Use only numpy and matplotlib. For trend lines, use numpy.polyfit instead of scipy.stats."""),
#         HumanMessage(content=prompt)
#     ]
    
#     response = llm.invoke(messages)
#     code = extract_code_from_response(response.content)
#     return code.strip()

# def execute_visualization_code(code: str):
#     """Execute the generated visualization code and return base64 image"""
#     try:
#         plt.clf()
#         plt.close('all')
#         namespace = {
#             'plt': plt,
#             'np': np,
#         }
        
#         print("Executing code:\n", code) #debugging
#         exec(code, namespace)
        
#         #base64 conversion
#         buf = io.BytesIO()
#         plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
#         buf.seek(0)
#         img_str = base64.b64encode(buf.getvalue()).decode('utf-8')

#         plt.close('all')
#         return img_str
    
#     except Exception as e:
#         raise Exception(f"Error executing visualization code: {str(e)}\nCode:\n{code}")

# def extract_chart_info(code: str):
#     """Extract chart information from the code"""
#     chart_info = {
#         "chart_type": "line",
#         "x_label": "X-axis",
#         "y_label": "Y-axis",
#         "title": "Visualization"
#     }
    
#     #chart type
#     if "plt.bar" in code:
#         chart_info["chart_type"] = "bar"
#     elif "plt.scatter" in code:
#         chart_info["chart_type"] = "scatter"
#     elif "plt.pie" in code:
#         chart_info["chart_type"] = "pie"
#     elif "plt.hist" in code:
#         chart_info["chart_type"] = "histogram"
    
#     title_match = re.search(r'plt\.title\(["\'](.+?)["\']\)', code)
#     if title_match:
#         chart_info["title"] = title_match.group(1)
    
#     xlabel_match = re.search(r'plt\.xlabel\(["\'](.+?)["\']\)', code)
#     if xlabel_match:
#         chart_info["x_label"] = xlabel_match.group(1)
    
#     ylabel_match = re.search(r'plt\.ylabel\(["\'](.+?)["\']\)', code)
#     if ylabel_match:
#         chart_info["y_label"] = ylabel_match.group(1)
    
#     return chart_info

# def answer_graph_question(session_id: str, question: str):
#     """Use Gemini to answer questions about the current graph"""
#     session = chat_sessions[session_id]
    
#     if not session.current_graph_code or not session.current_graph_data:
#         return "I don't have any graph data to analyze. Please generate a visualization first."

#     messages = [
#         SystemMessage(content="""You are a data visualization expert who can analyze and explain graphs. 
#         When answering questions about a graph, refer to the specific data points, trends, and visual elements in the graph.
#         Be precise and data-focused in your explanations.""")
#     ]
    
#     code_context = f"""
#     Here is the code and data for the current visualization:
#     ```python
#     {session.current_graph_code}
#     ```
    
#     The visualization has these properties:
#     - Chart type: {session.current_graph_data['chart_type']}
#     - Title: {session.current_graph_data['title']}
#     - X-axis: {session.current_graph_data['x_label']}
#     - Y-axis: {session.current_graph_data['y_label']}
#     """
    
#     messages.append(HumanMessage(content=code_context))
#     messages.append(AIMessage(content="I understand the visualization and its data. I'm ready to answer questions about it."))
#     messages.append(HumanMessage(content=question))
    
#     response = llm.invoke(messages)
#     return response.content

# @app.post("/sessions")
# async def create_session(request: NewSessionRequest):
#     """Create a new chat session"""
#     session_id = str(uuid.uuid4())
#     chat_sessions[session_id] = ChatSession(
#         id=session_id,
#         messages=[],
#         current_graph_code=None,
#         current_graph_data=None
#     )
#     return {"session_id": session_id}

# @app.get("/sessions")
# async def list_sessions():
#     """List all chat sessions"""
#     return {
#         "sessions": [
#             {"id": session_id, "message_count": len(session.messages)}
#             for session_id, session in chat_sessions.items()
#         ]
#     }

# @app.get("/sessions/{session_id}")
# async def get_session(session_id: str):
#     """Get a specific chat session"""
#     if session_id not in chat_sessions:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     return chat_sessions[session_id]

# @app.post("/chat")
# async def chat(request: ChatRequest):
#     """Process a chat message and generate a response"""
#     if not request.session_id or request.session_id not in chat_sessions: #if no session, creates a new one
#         session_id = str(uuid.uuid4())
#         chat_sessions[session_id] = ChatSession(
#             id=session_id,
#             messages=[],
#             current_graph_code=None,
#             current_graph_data=None
#         )
#     else:
#         session_id = request.session_id
    
#     session = chat_sessions[session_id]
    
#     user_message = ChatMessage(role="user", content=request.message) #add msg to history
#     session.messages.append(user_message)
    
#     # Check if this is a visualization request
#     is_visualization_request = any(keyword in request.message.lower() for keyword in 
#                                  ["create", "generate", "plot", "chart", "graph", "visualization", "visualize", "show me"])
    
#     try:
#         if is_visualization_request:
#             # Generate visualization
#             code = get_visualization_code(request.message)
#             base64_image = execute_visualization_code(code)
#             chart_info = extract_chart_info(code)
#             chart_info["title"] = chart_info.get("title", f"Visualization for: {request.message[:50]}...")
            
#             # Store graph data in session
#             session.current_graph_code = code
#             session.current_graph_data = chart_info
            
#             # Create assistant response
#             response_text = f"I've created the visualization based on your request. Here's what the graph shows:\n\n"
#             response_text += f"- This is a {chart_info['chart_type']} chart titled '{chart_info['title']}'\n"
#             response_text += f"- The x-axis represents {chart_info['x_label']}\n"
#             response_text += f"- The y-axis represents {chart_info['y_label']}\n\n"
#             response_text += "You can ask me specific questions about this visualization now."
            
#             assistant_message = ChatMessage(
#                 role="assistant", 
#                 content=response_text,
#                 image=base64_image
#             )
#         else:
#             # Answer a question about the existing graph
#             response_text = answer_graph_question(session_id, request.message)
#             assistant_message = ChatMessage(role="assistant", content=response_text)
        
#         # Add assistant response to history
#         session.messages.append(assistant_message)
        
#         return {
#             "session_id": session_id,
#             "response": assistant_message,
#             "chart_info": session.current_graph_data if is_visualization_request else None
#         }
    
#     except Exception as e:
#         error_message = f"Error processing your request: {str(e)}"
#         error_response = ChatMessage(role="assistant", content=error_message)
#         session.messages.append(error_response)
        
#         return {
#             "session_id": session_id,
#             "response": error_response,
#             "error": str(e)
#         }

# @app.delete("/sessions/{session_id}")
# async def delete_session(session_id: str):
#     """Delete a chat session"""
#     if session_id not in chat_sessions:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     del chat_sessions[session_id]
#     return {"status": "success", "message": f"Session {session_id} deleted"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



# import os
# import base64
# import io
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import google.generativeai as genai
# import matplotlib.pyplot as plt
# import numpy as np

# GEMINI_API_KEY = "AIzaSyDQcmV2pdj6Z2gSlCrWHfjojwyfOlBIv0Y"

# genai.configure(api_key=GEMINI_API_KEY)

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Model for chat request
# class ChatRequest(BaseModel):
#     query: str
#     chat_history: list = []

# # Model for chat response
# class ChatResponse(BaseModel):
#     response: str
#     graph: str = None  

# # Function to generate graph 
# def generate_graph(query):
    
#     try:
#         plt.figure(figsize=(10, 6))
#         x = np.linspace(0, 10, 100)
#         y = np.sin(x)
#         plt.plot(x, y)
#         plt.title(f"Graph for Query: {query}")
#         plt.xlabel('X Axis')
#         plt.ylabel('Y Axis')

#         buffer = io.BytesIO()
#         plt.savefig(buffer, format='png')
#         buffer.seek(0)
 
#         graph_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
#         plt.close()
        
#         return graph_base64
    
#     except Exception as e:
#         print(f"Graph generation error: {e}")
#         return None

# # Chat endpoint with comprehensive error handling
# @app.post("/chat")
# async def chat(request: ChatRequest):
#     try:
#         model = genai.GenerativeModel('gemini-1.5-flash')
#         response = model.generate_content(request.query)
#         graph_base64 = generate_graph(request.query)
        
#         return ChatResponse(
#             response=response.text, 
#             graph=graph_base64
#         )
    
#     except Exception as e:
#         import traceback
#         print(f"Detailed Error: {traceback.format_exc()}")
#         raise HTTPException(
#             status_code=500, 
#             detail=f"Internal server error: {str(e)}"
#         )

# import io
# import base64
# import re
# import ast
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Optional
# import matplotlib.pyplot as plt
# import google.generativeai as genai

# # ✅ Configure Gemini
# GEMINI_API_KEY = "AIzaSyDQcmV2pdj6Z2gSlCrWHfjojwyfOlBIv0Y"  # Replace this with your actual key
# genai.configure(api_key=GEMINI_API_KEY)

# # ✅ FastAPI App
# app = FastAPI()

# # ✅ Allow requests from frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ✅ Request/Response Models
# class Message(BaseModel):
#     role: str
#     content: str
#     graph_base64: Optional[str] = None

# class ChatRequest(BaseModel):
#     query: str
#     chat_history: List[Message] = []

# class ChatResponse(BaseModel):
#     response: str
#     graph_base64: Optional[str] = None

# # ✅ Graph Generation Function
# def generate_graph(chart_type: str, data):
#     plt.figure(figsize=(10, 6))

#     # Safely extract labels and values
#     if isinstance(data, dict):
#         labels = list(data.keys())
#         values = list(data.values())

#         # ✅ Flatten values if they're lists of lists or nested
#         if any(isinstance(v, list) for v in values):
#             values = [v[0] if isinstance(v, list) else v for v in values]

#     elif isinstance(data, list):
#         labels = list(range(len(data)))
#         values = data

#     else:
#         # Fallback dummy values
#         labels = ['A', 'B', 'C']
#         values = [10, 20, 30]

#     # ✅ Sanity check
#     if len(labels) != len(values):
#         raise ValueError("Labels and values length mismatch")

#     # Plot based on chart type
#     if chart_type == "bar":
#         plt.bar(labels, values)
#     elif chart_type == "line":
#         plt.plot(labels, values, marker='o')
#     elif chart_type == "pie":
#         plt.pie(values, labels=labels, autopct='%1.1f%%')
#     else:
#         plt.plot(labels, values)

#     plt.title(f"{chart_type.capitalize()} Chart")
#     buf = io.BytesIO()
#     plt.savefig(buf, format="png")
#     plt.close()
#     buf.seek(0)
#     return base64.b64encode(buf.read()).decode()


# # ✅ Main Chat Endpoint
# @app.post("/chat", response_model=ChatResponse)
# async def chat(request: ChatRequest):
#     try:
#         model = genai.GenerativeModel("gemini-1.5-flash")

#         # Build context-aware prompt
#         context_messages = "\n".join([
#             f"{msg.role.capitalize()}: {msg.content}" for msg in request.chat_history
#         ])

#         full_prompt = f"""
#         You are a powerful internet-connected data visualization assistant inside a chart-generating web app.

#         Your job is to:
#         - Understand the user's question (on any topic: sports, economics, business, health, politics, etc.)
#         - **Search the internet or use your latest knowledge** to provide real, accurate, and meaningful data
#         - Decide the **best graph type** (bar, pie, line, etc.)
#         - Respond **only** in the following format (no extra lines, no explanation):

#         Chart type: bar  
#         Data: {{ 'Label 1': value, 'Label 2': value, ... }}

#         Important Rules:
#         - Do NOT use made-up or dummy values like 0, 1, or 'example'
#         - Always provide real statistics or approximations based on trusted knowledge
#         - If recent data is not available, say: "Chart type: none" and "Data: none"

#         Context (chat history):
#         {context_messages}

#         User Query:
#         {request.query}
#         """



#         # Gemini reply
#         gemini_response = model.generate_content(full_prompt)
#         text = gemini_response.text or "No response."

#         # Extract chart type
#         type_match = re.search(r'chart type:\s*(\w+)', text, re.IGNORECASE)
#         chart_type = type_match.group(1).strip().lower() if type_match else "bar"

#         # Extract data block
#         data_match = re.search(r'data:\s*(\{.*?\}|\[.*?\])', text, re.IGNORECASE | re.DOTALL)
#         data_raw = data_match.group(1) if data_match else "[10, 20, 30]"
#         try:
#             data = ast.literal_eval(data_raw)
#         except Exception as eval_error:
#             print(f"Data parsing error: {eval_error}")
#             # Fallback if data is malformed
#             data = [10, 20, 30]
#             chart_type = "bar"


#         # Generate graph
#         graph_base64 = generate_graph(chart_type, data)

#         return ChatResponse(
#             response=text,
#             graph_base64=graph_base64
#         )

#     except Exception as e:
#         print(f"Server error: {e}")
#         raise HTTPException(status_code=500, detail="Internal Server Error")

import base64
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import matplotlib.pyplot as plt
import io
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import json

# Configure your API key
GEMINI_API_KEY = "AIzaSyDQcmV2pdj6Z2gSlCrWHfjojwyfOlBIv0Y"
genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str
    image: Optional[str] = None

class ChatRequest(BaseModel):
    messages: List[Message]
    query: str

class ChatResponse(BaseModel):
    response: str
    image: Optional[str] = None

# In-memory storage for chat history
chat_history = []

def generate_plot(data_description):
    """Generate a plot based on the AI's understanding of the data."""
    try:
        # This is a simplified placeholder for actual plot generation logic
        # In a real implementation, you would parse the AI's response to get data points
        
        # Example: Parse a simple line chart description
        # Format expected: "plot:line,x:[1,2,3,4],y:[10,20,15,25],title:Sample Plot"
        
        # For demonstration purposes, let's create a basic plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if "line" in data_description.lower():
            # Extract x and y data from the description
            import re
            x_match = re.search(r'x:\[([\d\s,.]+)\]', data_description)
            y_match = re.search(r'y:\[([\d\s,.]+)\]', data_description)
            title_match = re.search(r'title:([\w\s]+)', data_description)
            
            if x_match and y_match:
                x_data = [float(x) for x in x_match.group(1).split(',')]
                y_data = [float(y) for y in y_match.group(1).split(',')]
                title = title_match.group(1) if title_match else "Generated Plot"
                
                ax.plot(x_data, y_data, marker='o')
                ax.set_title(title)
                ax.grid(True)
            else:
                ax.text(0.5, 0.5, "Unable to parse plot data", ha='center', va='center')
        elif "bar" in data_description.lower():
            # Similar parsing for bar charts
            import re
            labels_match = re.search(r'labels:\[([\w\s,."\'-]+)\]', data_description)
            values_match = re.search(r'values:\[([\d\s,.]+)\]', data_description)
            title_match = re.search(r'title:([\w\s]+)', data_description)
            
            if labels_match and values_match:
                # Clean and parse labels
                labels_str = labels_match.group(1)
                labels = [label.strip().strip('"\'') for label in labels_str.split(',')]
                
                values = [float(v) for v in values_match.group(1).split(',')]
                title = title_match.group(1) if title_match else "Bar Chart"
                
                ax.bar(labels, values)
                ax.set_title(title)
            else:
                ax.text(0.5, 0.5, "Unable to parse bar chart data", ha='center', va='center')
        else:
            ax.text(0.5, 0.5, "Unknown chart type requested", ha='center', va='center')
        
        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        # Convert to base64 string
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return img_base64
    
    except Exception as e:
        print(f"Error generating plot: {str(e)}")
        # Create an error image
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}", ha='center', va='center')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return img_base64

def format_chat_history_for_gemini(messages):
    """Format chat history for Gemini API."""
    formatted_history = []
    
    for msg in messages:
        content = [{"text": msg.content}]
        
        # If there's an image in the message
        if msg.image:
            content.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": msg.image  # Base64 encoded image
                }
            })
            
        formatted_history.append({
            "role": msg.role,
            "parts": content
        })
    
    return formatted_history

@app.post("/plot", response_model=ChatResponse)
async def process_chat(request: ChatRequest):
    try:
        # Add the new user query to the chat history
        chat_history.append(Message(role="user", content=request.query))
        
        # Prepare conversation for Gemini
        gemini_messages = format_chat_history_for_gemini(request.messages)
        
        # Add system message to guide Gemini
        system_message = """
        You are a data visualization assistant. When asked for data visualization:
        1. Search for relevant data based on the query
        2. Format your response with clear data points for plotting
        3. Use format: plot:CHART_TYPE,x:[values],y:[values],title:TITLE for line charts
           or plot:bar,labels:[labels],values:[values],title:TITLE for bar charts
        4. If a follow-up question refers to previous visualizations, use that context
        5. Always provide a brief explanation with your visualization
        
        Keep responses focused on data visualization and related explanations.
        """
        
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Add system instruction
        chat = model.start_chat(history=[
            {"role": "user", "parts": [{"text": "System instructions: " + system_message}]},
            {"role": "model", "parts": [{"text": "I'll help with data visualization following those guidelines."}]}
        ])
        
        # Send conversation history and get response
        gemini_response = chat.send_message([{"text": request.query}])
        ai_response = gemini_response.text
        
        # Check if response contains plot data
        plot_data = None
        if "plot:" in ai_response:
            # Extract plot specification part
            plot_spec_start = ai_response.find("plot:")
            plot_spec_end = ai_response.find("\n", plot_spec_start)
            if plot_spec_end == -1:  # If no newline, take until the end
                plot_spec_end = len(ai_response)
            
            plot_specification = ai_response[plot_spec_start:plot_spec_end]
            plot_image = generate_plot(plot_specification)
            
            # Clean up the response by removing the plot specification
            ai_response = ai_response.replace(plot_specification, "")
            
            # Add image to chat history
            chat_history.append(Message(
                role="assistant", 
                content=ai_response,
                image=plot_image
            ))
            
            return ChatResponse(response=ai_response, image=plot_image)
        else:
            # No plot to generate, just return the text response
            chat_history.append(Message(role="assistant", content=ai_response))
            return ChatResponse(response=ai_response)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Data Visualization Chatbot API"}