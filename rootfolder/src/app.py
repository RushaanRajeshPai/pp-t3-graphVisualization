# import matplotlib.pyplot as plt
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.schema import SystemMessage, HumanMessage
# import os
# import sys
# import importlib

# # Set Google API key
# os.environ["GOOGLE_API_KEY"] = ""

# # Initialize LLM
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# # Make sure directories exist
# os.makedirs("src", exist_ok=True)
# os.makedirs("output", exist_ok=True)

# def check_package_availability():
#     """Check if optional visualization packages are available"""
#     available_packages = []
#     optional_packages = ["seaborn", "plotly", "squarify"]
    
#     for package in optional_packages:
#         try:
#             importlib.import_module(package)
#             available_packages.append(package)
#         except ImportError:
#             pass
    
#     return available_packages

# def generate_visualization(user_query):
#     """Generate a visualization based on user query"""
    
#     # Check available packages
#     available_packages = check_package_availability()
#     package_info = "Available optional packages: " + ", ".join(available_packages) if available_packages else "No optional packages available."
    
#     # Create a more robust prompt with clearer instructions
#     prompt = f"""
#     Write Python code to {user_query}. The code should:
#     1. Use matplotlib as the primary visualization library
#     2. Save the output as 'output/graph.png'
#     3. Use mock/sample data that's defined directly in the code (do not fetch external data)
#     4. Include proper indentation and formatting
#     5. Be completely self-contained with ALL necessary imports at the top
#     6. Handle directory creation safely using os.makedirs('output', exist_ok=True)
#     7. AVOID interactive features that require user input or window events
#     8. AVOID legends with clickable elements or hover events
#     9. If creating multiple subplots, use plt.subplots() and explicit axes
#     10. Include appropriate titles, labels, and color schemes
#     11. Use only standard libraries (matplotlib, numpy) - DO NOT use {', '.join(['squarify', 'plotly', 'bokeh', 'altair'])}
#     12. For complex layouts, prefer using subplots instead of advanced libraries
#     13. Ensure proper error handling, especially for directory creation
    
#     {package_info}
    
#     Example format:
#     ```python
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import os
    
#     # Create output directory if it doesn't exist
#     os.makedirs('output', exist_ok=True)
    
#     # Sample data
#     x = np.array([1, 2, 3, 4, 5])
#     y = np.array([10, 15, 7, 12, 9])
    
#     # Create plot
#     plt.figure(figsize=(10, 6))
#     plt.plot(x, y)
#     plt.title('Sample Chart')
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig('output/graph.png')
#     print('Visualization saved to output/graph.png')
#     ```
    
#     Example for multiple subplots:
#     ```python
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import os
    
#     # Create output directory if it doesn't exist
#     os.makedirs('output', exist_ok=True)
    
#     # Sample data
#     x = np.array([1, 2, 3, 4, 5])
#     y1 = np.array([10, 15, 7, 12, 9])
#     y2 = np.array([5, 8, 12, 9, 11])
    
#     # Create subplots
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
#     # First subplot
#     ax1.plot(x, y1)
#     ax1.set_title('Chart 1')
#     ax1.set_xlabel('X-axis')
#     ax1.set_ylabel('Y-axis')
    
#     # Second subplot
#     ax2.bar(x, y2)
#     ax2.set_title('Chart 2')
#     ax2.set_xlabel('X-axis')
#     ax2.set_ylabel('Y-axis')
    
#     plt.tight_layout()
#     plt.savefig('output/graph.png')
#     print('Visualization saved to output/graph.png')
#     ```
    
#     Important: Ensure all code has proper indentation and is executable Python code without syntax errors.
#     """
    
#     messages = [
#         SystemMessage(content="""You are an expert Python developer specializing in data visualization. Your code should be correct, properly formatted, and ready to execute.
        
#         Follow these strict guidelines:
#         1. Avoid any interactive features requiring user input or window events
#         2. Don't use external libraries that might not be installed 
#         3. Always include all necessary imports at the top
#         4. Never use interactive features of matplotlib (like mpl_connect)
#         5. Create static visualizations only
#         6. For complex visualizations, use multiple subplots rather than advanced libraries
#         7. Always use robust error handling"""),
#         HumanMessage(content=prompt)
#     ]
    
#     # Get response and extract code
#     response = llm.invoke(messages)  # Updated to use invoke() instead of __call__
#     full_response = response.content
    
#     # Try to extract just the code block from the response
#     import re
#     code_match = re.search(r'```python\s+(.*?)\s+```', full_response, re.DOTALL)
    
#     if code_match:
#         # Found a code block
#         generated_code = code_match.group(1)
#     else:
#         # No code block found, try to use the full response
#         generated_code = full_response
    
#     # Save the generated code
#     file_path = "src/generated_graph.py"
#     with open(file_path, "w") as f:
#         f.write(generated_code)
    
#     print(f"\nGenerated code saved to {file_path}")
    
#     # Execute the code
#     try:
#         print("\nExecuting the generated code...")
#         # Create a clean namespace to avoid variable conflicts
#         namespace = {'__name__': '__main__', 'plt': plt, 'os': os}
#         exec(generated_code, namespace)
#         print("Code executed successfully!")
        
#         if os.path.exists("output/graph.png"):
#             print("Visualization created successfully at 'output/graph.png'")
#             # In a notebook, you'd display the image here
#             # from IPython.display import Image
#             # return Image("output/graph.png")
#     except Exception as e:
#         print(f"Error executing code: {type(e).__name__}: {str(e)}")
#         print("\nHere's the problematic code:")
#         lines = generated_code.split('\n')
#         for i, line in enumerate(lines):
#             print(f"{i+1}: {line}")

# # Interactive loop for user queries
# while True:
#     user_input = input("\nEnter your visualization query (or 'exit' to quit): ")
    
#     if user_input.lower() == 'exit':
#         print("Exiting program.")
#         break
    
#     generate_visualization(user_input)


# import io
# import base64
# import json
# import os
# from typing import Optional

# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import requests
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import google.generativeai as genai

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDQcmV2pdj6Z2gSlCrWHfjojwyfOlBIv0Y")
# genai.configure(api_key=GEMINI_API_KEY)

# class PlotRequest(BaseModel):
#     query: str

# def generate_data_from_gemini(query: str):
#     """Use Gemini to interpret the query and generate appropriate data."""
#     prompt = f"""
#     I need to create a visualization for: "{query}"
    
#     Please analyze this request and provide:
#     1. A suitable dataset for this visualization (as a JSON array of objects)
#     2. The type of chart that would be most appropriate (line, bar, scatter, pie, etc.)
#     3. X and Y axis labels
#     4. A title for the chart
    
#     Format your response as valid JSON like this:
#     {{
#         "data": [{{x: value, y: value}}, ...],
#         "chart_type": "line/bar/scatter/pie",
#         "x_label": "X-axis label",
#         "y_label": "Y-axis label",
#         "title": "Chart title"
#     }}
    
#     Only provide the JSON output, nothing else.
#     """
    
#     model = genai.GenerativeModel('gemini-1.5-flash')
#     response = model.generate_content(prompt)
    
#     try:
#         json_str = response.text
#         if "```json" in json_str:
#             json_str = json_str.split("```json")[1].split("```")[0].strip()
#         elif "```" in json_str:
#             json_str = json_str.split("```")[1].split("```")[0].strip()
        
#         result = json.loads(json_str)
#         return result
#     except (json.JSONDecodeError, IndexError) as e:
#         raise HTTPException(status_code=500, detail=f"Failed to parse Gemini response: {str(e)}")

# def create_visualization(data_config):
#     """Create a visualization based on the data and configuration."""
#     plt.figure(figsize=(10, 6))
    
#     data = data_config["data"]
#     chart_type = data_config["chart_type"].lower()
    
#     if isinstance(data, list) and all(isinstance(item, dict) for item in data):
#         #Extracting x and y values
#         x_values = [item.get("x") for item in data]
#         y_values = [item.get("y") for item in data]
        
#         if chart_type == "line":
#             plt.plot(x_values, y_values, marker='o')
#         elif chart_type == "bar":
#             plt.bar(x_values, y_values)
#         elif chart_type == "scatter":
#             plt.scatter(x_values, y_values)
#         elif chart_type == "pie":
#             plt.pie(y_values, labels=x_values, autopct='%1.1f%%')
#         else:
#             # Default to line chart
#             plt.plot(x_values, y_values, marker='o')
    
#     plt.xlabel(data_config.get("x_label", ""))
#     plt.ylabel(data_config.get("y_label", ""))
#     plt.title(data_config.get("title", ""))
#     plt.grid(True)
#     plt.tight_layout()
    
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
    
#     #Encoding the image as base64
#     # img_str = base64.b64encode(buf.read()).decode('utf-8')
#     img_str = base64.b64encode(buf.read())
#     plt.close()
    
#     return img_str

# @app.post("/plot")
# async def generate_plot(request: PlotRequest):
#     """Generate a plot based on the user's query."""
#     try:
#         #Generating data using Gemini
#         data_config = generate_data_from_gemini(request.query)
        
#         #Create visualization and convert to base64
#         base64_image = create_visualization(data_config)
        
#         # Return the base64 encoded image
#         return {"image": base64_image, "config": data_config}
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



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

# app = FastAPI()

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

# # Make sure directories exist
# os.makedirs("src", exist_ok=True)

# def check_package_availability():
#     """Check if optional visualization packages are available"""
#     available_packages = []
#     optional_packages = ["seaborn", "plotly", "squarify"]
    
#     for package in optional_packages:
#         try:
#             importlib.import_module(package)
#             available_packages.append(package)
#         except ImportError:
#             pass
    
#     return available_packages

# def plot_to_base64():
#     """Convert the current matplotlib plot to a base64 string"""
#     # Create a bytes buffer for the image
#     buf = io.BytesIO()
    
#     # Save the plot to the buffer
#     plt.savefig(buf, format='png')
#     buf.seek(0)
    
#     # Convert to base64
#     img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    
#     # Close the plot to free memory
#     plt.close()
    
#     return img_str

# def create_visualization(data_config):
#     """Create visualization based on the data configuration and return as base64"""
#     try:
#         # Execute the generated code from data_config
#         namespace = {'__name__': '__main__', 'plt': plt, 'os': os, 'np': __import__('numpy')}
        
#         # Make sure the code doesn't actually save the file but returns the plot object
#         modified_code = data_config["code"].replace("plt.savefig", "# plt.savefig")
        
#         # Execute the code
#         exec(modified_code, namespace)
        
#         # Convert the plot to base64
#         base64_image = plot_to_base64()
        
#         return base64_image
        
#     except Exception as e:
#         raise Exception(f"Error creating visualization: {str(e)}")

# def generate_data_from_gemini(user_query):
#     """Generate visualization code based on user query using Gemini"""
    
#     # Check available packages
#     available_packages = check_package_availability()
#     package_info = "Available optional packages: " + ", ".join(available_packages) if available_packages else "No optional packages available."
    
#     # Create a more robust prompt with clearer instructions
#     prompt = f"""
#     Write Python code to {user_query}. The code should:
#     1. Use matplotlib as the primary visualization library
#     2. Do NOT save the output to a file - we'll handle that later
#     3. Use mock/sample data that's defined directly in the code (do not fetch external data)
#     4. Include proper indentation and formatting
#     5. Be completely self-contained with ALL necessary imports at the top
#     6. AVOID interactive features that require user input or window events
#     7. AVOID legends with clickable elements or hover events
#     8. If creating multiple subplots, use plt.subplots() and explicit axes
#     9. Include appropriate titles, labels, and color schemes
#     10. Use only standard libraries (matplotlib, numpy) - DO NOT use {', '.join(['squarify', 'plotly', 'bokeh', 'altair'])}
#     11. For complex layouts, prefer using subplots instead of advanced libraries
    
#     {package_info}
    
#     Example format:
#     ```python
#     import matplotlib.pyplot as plt
#     import numpy as np
    
#     # Sample data
#     x = np.array([1, 2, 3, 4, 5])
#     y = np.array([10, 15, 7, 12, 9])
    
#     # Create plot
#     plt.figure(figsize=(10, 6))
#     plt.plot(x, y)
#     plt.title('Sample Chart')
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     ```
    
#     Important: Ensure all code has proper indentation and is executable Python code without syntax errors.
#     """
    
#     messages = [
#         SystemMessage(content="""You are an expert Python developer specializing in data visualization. Your code should be correct, properly formatted, and ready to execute.
        
#         Follow these strict guidelines:
#         1. Avoid any interactive features requiring user input or window events
#         2. Don't use external libraries that might not be installed 
#         3. Always include all necessary imports at the top
#         4. Never use interactive features of matplotlib (like mpl_connect)
#         5. Create static visualizations only
#         6. For complex visualizations, use multiple subplots rather than advanced libraries
#         7. DO NOT include code to save the visualization to a file - that will be handled separately"""),
#         HumanMessage(content=prompt)
#     ]
    
#     # Get response and extract code
#     response = llm.invoke(messages)
#     full_response = response.content
    
#     # Try to extract just the code block from the response
#     import re
#     code_match = re.search(r'```python\s+(.*?)\s+```', full_response, re.DOTALL)
    
#     if code_match:
#         # Found a code block
#         generated_code = code_match.group(1)
#     else:
#         # No code block found, try to use the full response
#         generated_code = full_response
    
#     # Save the generated code
#     file_path = "src/generated_graph.py"
#     with open(file_path, "w") as f:
#         f.write(generated_code)
    
#     # Return the data configuration
#     return {
#         "code": generated_code,
#         "query": user_query
#     }

# @app.post("/plot")
# async def generate_plot(request: PlotRequest):
#     """Generate a plot based on the user's query."""
#     try:
#         # Generating data using Gemini
#         data_config = generate_data_from_gemini(request.query)
        
#         # Create visualization and convert to base64
#         base64_image = create_visualization(data_config)
        
#         # Return the base64 encoded image
#         return {"image": base64_image, "config": data_config}
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

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

def extract_code_from_response(response_text: str) -> str:
    """Extract Python code from Gemini's response"""
    # Try to find code between Python code blocks
    code_match = re.search(r'python\n(.*?)\n', response_text, re.DOTALL)
    if code_match:
        return code_match.group(1)
    
    # If no Python blocks found, try any code blocks
    code_match = re.search(r'(.*?)', response_text, re.DOTALL)
    if code_match:
        return code_match.group(1)
    
    # If no code blocks found, return the entire response
    return response_text

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