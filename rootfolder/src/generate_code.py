import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
import os
import sys
import importlib

# Set Google API key
os.environ["GOOGLE_API_KEY"] = ""

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Make sure directories exist
os.makedirs("src", exist_ok=True)
os.makedirs("output", exist_ok=True)

def check_package_availability():
    """Check if optional visualization packages are available"""
    available_packages = []
    optional_packages = ["seaborn", "plotly", "squarify"]
    
    for package in optional_packages:
        try:
            importlib.import_module(package)
            available_packages.append(package)
        except ImportError:
            pass
    
    return available_packages

def generate_visualization(user_query):
    """Generate a visualization based on user query"""
    
    # Check available packages
    available_packages = check_package_availability()
    package_info = "Available optional packages: " + ", ".join(available_packages) if available_packages else "No optional packages available."
    
    # Create a more robust prompt with clearer instructions
    prompt = f"""
    Write Python code to {user_query}. The code should:
    1. Use matplotlib as the primary visualization library
    2. Save the output as 'output/graph.png'
    3. Use mock/sample data that's defined directly in the code (do not fetch external data)
    4. Include proper indentation and formatting
    5. Be completely self-contained with ALL necessary imports at the top
    6. Handle directory creation safely using os.makedirs('output', exist_ok=True)
    7. AVOID interactive features that require user input or window events
    8. AVOID legends with clickable elements or hover events
    9. If creating multiple subplots, use plt.subplots() and explicit axes
    10. Include appropriate titles, labels, and color schemes
    11. Use only standard libraries (matplotlib, numpy) - DO NOT use {', '.join(['squarify', 'plotly', 'bokeh', 'altair'])}
    12. For complex layouts, prefer using subplots instead of advanced libraries
    13. Ensure proper error handling, especially for directory creation
    
    {package_info}
    
    Example format:
    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Sample data
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([10, 15, 7, 12, 9])
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title('Sample Chart')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/graph.png')
    print('Visualization saved to output/graph.png')
    ```
    
    Example for multiple subplots:
    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Sample data
    x = np.array([1, 2, 3, 4, 5])
    y1 = np.array([10, 15, 7, 12, 9])
    y2 = np.array([5, 8, 12, 9, 11])
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # First subplot
    ax1.plot(x, y1)
    ax1.set_title('Chart 1')
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    
    # Second subplot
    ax2.bar(x, y2)
    ax2.set_title('Chart 2')
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Y-axis')
    
    plt.tight_layout()
    plt.savefig('output/graph.png')
    print('Visualization saved to output/graph.png')
    ```
    
    Important: Ensure all code has proper indentation and is executable Python code without syntax errors.
    """
    
    messages = [
        SystemMessage(content="""You are an expert Python developer specializing in data visualization. Your code should be correct, properly formatted, and ready to execute.
        
        Follow these strict guidelines:
        1. Avoid any interactive features requiring user input or window events
        2. Don't use external libraries that might not be installed 
        3. Always include all necessary imports at the top
        4. Never use interactive features of matplotlib (like mpl_connect)
        5. Create static visualizations only
        6. For complex visualizations, use multiple subplots rather than advanced libraries
        7. Always use robust error handling"""),
        HumanMessage(content=prompt)
    ]
    
    # Get response and extract code
    response = llm.invoke(messages)  # Updated to use invoke() instead of __call__
    full_response = response.content
    
    # Try to extract just the code block from the response
    import re
    code_match = re.search(r'```python\s+(.*?)\s+```', full_response, re.DOTALL)
    
    if code_match:
        # Found a code block
        generated_code = code_match.group(1)
    else:
        # No code block found, try to use the full response
        generated_code = full_response
    
    # Save the generated code
    file_path = "src/generated_graph.py"
    with open(file_path, "w") as f:
        f.write(generated_code)
    
    print(f"\nGenerated code saved to {file_path}")
    
    # Execute the code
    try:
        print("\nExecuting the generated code...")
        # Create a clean namespace to avoid variable conflicts
        namespace = {'__name__': '__main__', 'plt': plt, 'os': os}
        exec(generated_code, namespace)
        print("Code executed successfully!")
        
        if os.path.exists("output/graph.png"):
            print("Visualization created successfully at 'output/graph.png'")
            # In a notebook, you'd display the image here
            # from IPython.display import Image
            # return Image("output/graph.png")
    except Exception as e:
        print(f"Error executing code: {type(e).__name__}: {str(e)}")
        print("\nHere's the problematic code:")
        lines = generated_code.split('\n')
        for i, line in enumerate(lines):
            print(f"{i+1}: {line}")

# Interactive loop for user queries
while True:
    user_input = input("\nEnter your visualization query (or 'exit' to quit): ")
    
    if user_input.lower() == 'exit':
        print("Exiting program.")
        break
    
    generate_visualization(user_input)