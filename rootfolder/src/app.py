import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
import os
import sys
import importlib
import io
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import re

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PlotRequest(BaseModel):
    query: str

# Set Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDQcmV2pdj6Z2gSlCrWHfjojwyfOlBIv0Y"

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# def extract_code_from_response(response_text: str) -> str:
    
    
#     code_match = re.search(r'python\n(.*?)\n', response_text, re.DOTALL)
#     if code_match:
#         return code_match.group(1)
    
    
#     code_match = re.search(r'(.*?)', response_text, re.DOTALL)
#     if code_match:
#         return code_match.group(1)
    
    
#     return response_text

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
    # but print a warning for debugging
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

    Example format:
    python
    import matplotlib.pyplot as plt
    import numpy as np

    # Create figure
    plt.figure(figsize=(15, 5))

    # For trend lines, use:
    coefficients = np.polyfit(x, y, 1)
    trend_line = coefficients[0] * x + coefficients[1]
    
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
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        
        # Create namespace with required imports
        namespace = {
            'plt': plt,
            'np': np,
        }
        
        # Print code for debugging
        print("Executing code:\n", code)
        
        # Execute the code
        exec(code, namespace)
        
        # Convert plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Clean up
        plt.close('all')
        
        return img_str
    except Exception as e:
        raise Exception(f"Error executing visualization code: {str(e)}\nCode:\n{code}")

@app.post("/plot")
async def generate_plot(request: PlotRequest):
    """Generate a plot based on the user's query."""
    try:
        # Get visualization code from Gemini
        code = get_visualization_code(request.query)
        
        # Execute the code and get base64 image
        base64_image = execute_visualization_code(code)
        
        # Extract chart type from code (simple heuristic)
        chart_type = "line"
        if "plt.bar" in code:
            chart_type = "bar"
        elif "plt.scatter" in code:
            chart_type = "scatter"
        elif "plt.pie" in code:
            chart_type = "pie"
        
        # Return the response
        return {
            "image": base64_image,
            "config": {
                "title": f"Visualization for: {request.query}",
                "chart_type": chart_type,
                "x_label": "X-axis",
                "y_label": "Y-axis"
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)