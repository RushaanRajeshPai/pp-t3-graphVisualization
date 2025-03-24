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
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import os
import io
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import re
import uuid
from typing import List, Dict, Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str  
    content: str
    image: Optional[str] = None

class ChatSession(BaseModel):
    id: str
    messages: List[ChatMessage]
    current_graph_code: Optional[str] = None
    current_graph_data: Optional[Dict] = None

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class NewSessionRequest(BaseModel):
    name: Optional[str] = None

chat_sessions: Dict[str, ChatSession] = {} #chat sessions stored in dict format

os.environ["GOOGLE_API_KEY"] = "AIzaSyDQcmV2pdj6Z2gSlCrWHfjojwyfOlBIv0Y"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

def extract_code_from_response(response_text: str) -> str:
    # Look for code blocks with ```python ... ``` format
    code_block_match = re.search(r'```python\s+(.*?)\s+```', response_text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()
    
    # If no code block found, look for code sections with "python" marker
    code_section_match = re.search(r'python\s+(.*?)(?:\n\n|$)', response_text, re.DOTALL)
    if code_section_match:
        return code_section_match.group(1).strip()
    
    # If still nothing found, just return the whole response as a fallback
    print("WARNING: Could not extract code block from response, using full response.")
    return response_text.strip()

def get_visualization_code(query: str):
    """Get visualization code from Gemini"""
    prompt = f"""
    Generate Python code for the following visualization request: {query}

    Requirements:
    1. Use ONLY matplotlib and numpy (DO NOT use scipy or other libraries)
    2. For trend lines, use numpy.polyfit instead of scipy.stats
    3. Include all necessary imports at the top
    4. Create sample data arrays that MUST have exactly the same length
    5. Add proper title, labels, and grid
    6. Make the visualization clear and professional
    7. DO NOT include plt.show() or plt.savefig()
    8. Use plt.figure(figsize=(15, 5)) for wide multi-panel plots
    9. SAVE THE ACTUAL DATA VALUES you generate in comments at the bottom of the code for later reference
    """
    
    messages = [
        SystemMessage(content="""You are a Python data visualization expert. Generate only executable matplotlib code.
        IMPORTANT: Use only numpy and matplotlib. For trend lines, use numpy.polyfit instead of scipy.stats."""),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    code = extract_code_from_response(response.content)
    return code.strip()

def execute_visualization_code(code: str):
    """Execute the generated visualization code and return base64 image"""
    try:
        plt.clf()
        plt.close('all')
        namespace = {
            'plt': plt,
            'np': np,
        }
        
        print("Executing code:\n", code) #debugging
        exec(code, namespace)
        
        #base64 conversion
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')

        plt.close('all')
        return img_str
    
    except Exception as e:
        raise Exception(f"Error executing visualization code: {str(e)}\nCode:\n{code}")

def extract_chart_info(code: str):
    """Extract chart information from the code"""
    chart_info = {
        "chart_type": "line",
        "x_label": "X-axis",
        "y_label": "Y-axis",
        "title": "Visualization"
    }
    
    #chart type
    if "plt.bar" in code:
        chart_info["chart_type"] = "bar"
    elif "plt.scatter" in code:
        chart_info["chart_type"] = "scatter"
    elif "plt.pie" in code:
        chart_info["chart_type"] = "pie"
    elif "plt.hist" in code:
        chart_info["chart_type"] = "histogram"
    
    title_match = re.search(r'plt\.title\(["\'](.+?)["\']\)', code)
    if title_match:
        chart_info["title"] = title_match.group(1)
    
    xlabel_match = re.search(r'plt\.xlabel\(["\'](.+?)["\']\)', code)
    if xlabel_match:
        chart_info["x_label"] = xlabel_match.group(1)
    
    ylabel_match = re.search(r'plt\.ylabel\(["\'](.+?)["\']\)', code)
    if ylabel_match:
        chart_info["y_label"] = ylabel_match.group(1)
    
    return chart_info

def answer_graph_question(session_id: str, question: str):
    """Use Gemini to answer questions about the current graph"""
    session = chat_sessions[session_id]
    
    if not session.current_graph_code or not session.current_graph_data:
        return "I don't have any graph data to analyze. Please generate a visualization first."

    messages = [
        SystemMessage(content="""You are a data visualization expert who can analyze and explain graphs. 
        When answering questions about a graph, refer to the specific data points, trends, and visual elements in the graph.
        Be precise and data-focused in your explanations.""")
    ]
    
    code_context = f"""
    Here is the code and data for the current visualization:
    ```python
    {session.current_graph_code}
    ```
    
    The visualization has these properties:
    - Chart type: {session.current_graph_data['chart_type']}
    - Title: {session.current_graph_data['title']}
    - X-axis: {session.current_graph_data['x_label']}
    - Y-axis: {session.current_graph_data['y_label']}
    """
    
    messages.append(HumanMessage(content=code_context))
    messages.append(AIMessage(content="I understand the visualization and its data. I'm ready to answer questions about it."))
    messages.append(HumanMessage(content=question))
    
    response = llm.invoke(messages)
    return response.content

@app.post("/sessions")
async def create_session(request: NewSessionRequest):
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = ChatSession(
        id=session_id,
        messages=[],
        current_graph_code=None,
        current_graph_data=None
    )
    return {"session_id": session_id}

@app.get("/sessions")
async def list_sessions():
    """List all chat sessions"""
    return {
        "sessions": [
            {"id": session_id, "message_count": len(session.messages)}
            for session_id, session in chat_sessions.items()
        ]
    }

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a specific chat session"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return chat_sessions[session_id]

@app.post("/chat")
async def chat(request: ChatRequest):
    """Process a chat message and generate a response"""
    if not request.session_id or request.session_id not in chat_sessions: #if no session, creates a new one
        session_id = str(uuid.uuid4())
        chat_sessions[session_id] = ChatSession(
            id=session_id,
            messages=[],
            current_graph_code=None,
            current_graph_data=None
        )
    else:
        session_id = request.session_id
    
    session = chat_sessions[session_id]
    
    user_message = ChatMessage(role="user", content=request.message) #add msg to history
    session.messages.append(user_message)
    
    # Check if this is a visualization request
    is_visualization_request = any(keyword in request.message.lower() for keyword in 
                                 ["create", "generate", "plot", "chart", "graph", "visualization", "visualize", "show me"])
    
    try:
        if is_visualization_request:
            # Generate visualization
            code = get_visualization_code(request.message)
            base64_image = execute_visualization_code(code)
            chart_info = extract_chart_info(code)
            chart_info["title"] = chart_info.get("title", f"Visualization for: {request.message[:50]}...")
            
            # Store graph data in session
            session.current_graph_code = code
            session.current_graph_data = chart_info
            
            # Create assistant response
            response_text = f"I've created the visualization based on your request. Here's what the graph shows:\n\n"
            response_text += f"- This is a {chart_info['chart_type']} chart titled '{chart_info['title']}'\n"
            response_text += f"- The x-axis represents {chart_info['x_label']}\n"
            response_text += f"- The y-axis represents {chart_info['y_label']}\n\n"
            response_text += "You can ask me specific questions about this visualization now."
            
            assistant_message = ChatMessage(
                role="assistant", 
                content=response_text,
                image=base64_image
            )
        else:
            # Answer a question about the existing graph
            response_text = answer_graph_question(session_id, request.message)
            assistant_message = ChatMessage(role="assistant", content=response_text)
        
        # Add assistant response to history
        session.messages.append(assistant_message)
        
        return {
            "session_id": session_id,
            "response": assistant_message,
            "chart_info": session.current_graph_data if is_visualization_request else None
        }
    
    except Exception as e:
        error_message = f"Error processing your request: {str(e)}"
        error_response = ChatMessage(role="assistant", content=error_message)
        session.messages.append(error_response)
        
        return {
            "session_id": session_id,
            "response": error_response,
            "error": str(e)
        }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del chat_sessions[session_id]
    return {"status": "success", "message": f"Session {session_id} deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)