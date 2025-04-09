import base64
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import matplotlib.pyplot as plt
import io
import google.generativeai as genai
from typing import List, Dict, Optional
import json
import numpy as np
from matplotlib.colors import to_rgba
import re
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBOKz-eFNXkPDBwTzoo-XCoILuwQqXtS84")
genai.configure(api_key=GEMINI_API_KEY)
app = FastAPI()
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
    metadata: Optional[Dict] = None
class ChatRequest(BaseModel):  #frontend to backend
    messages: List[Message]
    query: str
class ChatResponse(BaseModel): #backend to frontend
    response: str
    image: Optional[str] = None
    metadata: Optional[Dict] = None

chat_history = []

def generate_plot(data_description):
    """Generate a plot based on the AI's description, supporting multiple chart types."""
    try:
        type_match = re.search(r'plot:(\w+)', data_description.lower())
        if not type_match:
            raise ValueError("Chart type not specified in the format")
        
        chart_type = type_match.group(1).lower()
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_data = {"type": chart_type}

        patterns = {
            "x": r'x:\[([\d\s,.]+)\]',
            "y": r'y:\[([\d\s,.]+)\]',
            "labels": r'labels:\[([\w\s,."\'-]+)\]',
            "values": r'values:\[([\d\s,.]+)\]',
            "title": r'title:([\w\s]+)',
            "colors": r'colors:\[([\w\s,.#\'"-]+)\]',
            "sizes": r'sizes:\[([\d\s,.]+)\]',
            "categories": r'categories:\[([\w\s,."\'-]+)\]',
            "series": r'series:\[([\w\s,."\'-]+)\]'
        }

        extracted_data = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, data_description)
            if match:
                if key in ["x", "y", "values", "sizes"]:
                    extracted_data[key] = [float(x) for x in match.group(1).split(',')]
                elif key in ["labels", "categories", "series", "colors"]:
                    data_str = match.group(1)
                    extracted_data[key] = [item.strip().strip('"\'') for item in data_str.split(',')]
                else:
                    extracted_data[key] = match.group(1).strip()
        
        title = extracted_data.get("title", f"{chart_type.capitalize()} Chart")
        plot_data.update(extracted_data)
        
        # Create the appropriate chart based on type
        if chart_type == "line":
            if "x" in extracted_data and "y" in extracted_data:
                line = ax.plot(extracted_data["x"], extracted_data["y"], marker='o')
                
                # Apply colors if specified
                if "colors" in extracted_data and len(extracted_data["colors"]) > 0:
                    line[0].set_color(extracted_data["colors"][0])
                
                ax.set_title(title)
                ax.grid(True)
            else:
                raise ValueError("Line chart requires x and y data")
                
        elif chart_type == "bar":
            if "labels" in extracted_data and "values" in extracted_data:
                labels = extracted_data["labels"]
                values = extracted_data["values"]
                
                # Use specified colors or default
                colors = extracted_data.get("colors", None)
                bars = ax.bar(labels, values, color=colors)
                
                ax.set_title(title)
                # Rotate labels if there are many
                if len(labels) > 5:
                    plt.xticks(rotation=45, ha='right')
            else:
                raise ValueError("Bar chart requires labels and values")
                
        elif chart_type == "pie":
            if "labels" in extracted_data and "values" in extracted_data:
                labels = extracted_data["labels"]
                values = extracted_data["values"]
                
                # Use specified colors or default
                colors = extracted_data.get("colors", None)
                
                # Create pie chart
                ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors)
                ax.set_title(title)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            else:
                raise ValueError("Pie chart requires labels and values")
                
        elif chart_type == "scatter":
            if "x" in extracted_data and "y" in extracted_data:
                x = extracted_data["x"]
                y = extracted_data["y"]
                
                # Optional size parameter
                sizes = extracted_data.get("sizes", [50] * len(x))
                colors = extracted_data.get("colors", ["blue"] * len(x))
                
                # Handle case where we have fewer sizes or colors than points
                if len(sizes) < len(x):
                    sizes = sizes * (len(x) // len(sizes) + 1)
                    sizes = sizes[:len(x)]
                if len(colors) < len(x):
                    colors = colors * (len(x) // len(colors) + 1)
                    colors = colors[:len(x)]
                
                scatter = ax.scatter(x, y, s=sizes, c=colors, alpha=0.7)
                ax.set_title(title)
                ax.grid(True)
            else:
                raise ValueError("Scatter chart requires x and y data")
                
        elif chart_type == "histogram":
            if "values" in extracted_data:
                values = extracted_data["values"]
                bins = extracted_data.get("bins", 10)  # Default 10 bins if not specified
                
                # Use specified colors or default
                color = extracted_data.get("colors", ["blue"])[0] if "colors" in extracted_data else "blue"
                
                hist = ax.hist(values, bins=bins, color=color, alpha=0.7, edgecolor="black")
                ax.set_title(title)
                ax.grid(True)
            else:
                raise ValueError("Histogram requires values data")
                
        elif chart_type == "area":
            if "x" in extracted_data and "y" in extracted_data:
                x = extracted_data["x"]
                y = extracted_data["y"]
                
                # Use specified colors or default
                color = extracted_data.get("colors", ["blue"])[0] if "colors" in extracted_data else "blue"
                
                ax.fill_between(x, y, color=color, alpha=0.5)
                ax.plot(x, y, color=color)
                ax.set_title(title)
                ax.grid(True)
            else:
                raise ValueError("Area chart requires x and y data")
                
        elif chart_type == "heatmap":
            if "values" in extracted_data:
                # Assuming values is a flattened matrix
                values = extracted_data["values"]
                
                # Try to determine matrix dimensions
                size = int(np.sqrt(len(values)))
                if size * size != len(values):
                    # If not a perfect square, try to use provided dimensions
                    x_len = len(extracted_data.get("x", []))
                    y_len = len(extracted_data.get("y", []))
                    
                    if x_len * y_len == len(values):
                        size_x, size_y = x_len, y_len
                    else:
                        raise ValueError("Cannot determine heatmap dimensions. Please provide square data or explicit dimensions")
                else:
                    size_x, size_y = size, size
                
                # Reshape into a 2D matrix
                matrix = np.array(values).reshape(size_y, size_x)
                
                # Create heatmap
                heatmap = ax.imshow(matrix, cmap='viridis')
                fig.colorbar(heatmap, ax=ax)
                
                # Add labels if provided
                if "x" in extracted_data and len(extracted_data["x"]) == size_x:
                    plt.xticks(range(size_x), extracted_data["x"])
                if "y" in extracted_data and len(extracted_data["y"]) == size_y:
                    plt.yticks(range(size_y), extracted_data["y"])
                
                ax.set_title(title)
            else:
                raise ValueError("Heatmap requires values data")
        
        elif chart_type == "boxplot":
            if "values" in extracted_data:
                # For single boxplot
                ax.boxplot(extracted_data["values"])
                ax.set_title(title)
                
                # Add categories if provided
                if "categories" in extracted_data:
                    plt.xticks([1], [extracted_data["categories"][0]])
            elif "categories" in extracted_data and len(extracted_data["categories"]) > 0:
                # For multiple boxplots, we need data for each category
                # Example: categories=["A","B"], values_A=[1,2,3], values_B=[4,5,6]
                data = []
                categories = extracted_data["categories"]
                
                for category in categories:
                    cat_key = f"values_{category}"
                    cat_match = re.search(f'{cat_key}:\[([\d\s,.]+)\]', data_description)
                    
                    if cat_match:
                        cat_data = [float(x) for x in cat_match.group(1).split(',')]
                        data.append(cat_data)
                    else:
                        raise ValueError(f"Missing data for category {category}")
                
                ax.boxplot(data)
                ax.set_title(title)
                plt.xticks(range(1, len(categories) + 1), categories)
            else:
                raise ValueError("Boxplot requires values data or category-specific data")
                
        elif chart_type == "radar" or chart_type == "spider":
            if "categories" in extracted_data and "values" in extracted_data:
                categories = extracted_data["categories"]
                values = extracted_data["values"]
                
                if len(categories) != len(values):
                    raise ValueError("Number of categories must match number of values for radar chart")
                
                # Create a figure with polar projection
                plt.close(fig)  # Close the previous figure
                fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
                
                # Number of variables
                N = len(categories)
                
                # What angles to place each category
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # Close the loop
                
                # Values need to close the loop as well
                values += values[:1]
                
                # Draw the chart
                ax.plot(angles, values, linewidth=2, linestyle='solid')
                ax.fill(angles, values, alpha=0.25)
                
                # Fix axis to go in the right order and start at 12 o'clock
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                
                # Draw category labels on the axes
                plt.xticks(angles[:-1], categories)
                
                ax.set_title(title)
            else:
                raise ValueError("Radar chart requires categories and values")
                
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plt.close(fig)
        
        # Convert to base64 string
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return img_base64, plot_data
    
    except Exception as e:
        print(f"Error generating plot: {str(e)}")
        # Create an error image
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}", ha='center', va='center')
        fig.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return img_base64, {"error": str(e)}

def format_chat_history_for_gemini(messages):
    """Format chat history for Gemini API."""
    formatted_history = []
    
    for msg in messages:
        context_text = msg.content
        if msg.metadata:
            context_text += f"\nContext data: {json.dumps(msg.metadata)}"
        
        content = [{"text": context_text}]
        
        if msg.image:
            content.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": msg.image  #base64 string format
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
        previous_context = [m.metadata for m in request.messages if m.metadata]
        chat_history.append(Message(
            role="user", 
            content=request.query,
            metadata={"previous_context": previous_context} if previous_context else None
        ))
        
        # Prepare conversation for Gemini
        gemini_messages = format_chat_history_for_gemini(request.messages)
        
        system_message = """
        You are an expert data visualization assistant with access to a wide range of data. Your primary function is to create visualizations from data. For EVERY query:

        1. ALWAYS generate sample data that reasonably represents what the user is asking for, even if you don't have real-time or specific data access.

        2. FORMAT YOUR RESPONSE properly with:
        - An introduction explaining the visualization 
        - The plot specification in machine-parseable format
        - Additional context/insights about the data

        3. For LINE CHARTS use:
        plot:line,x:[1,2,3,...],y:[10,20,30,...],title:Your Title

        4. For BAR CHARTS use:
        plot:bar,labels:["Label1","Label2",...],values:[10,20,...],title:Your Title

        5. For PIE CHARTS use:
        plot:pie,labels:["Label1","Label2",...],values:[30,70,...],title:Your Title

        6. For SCATTER PLOTS use:
        plot:scatter,x:[1,2,3,...],y:[10,20,30,...],title:Your Title

        7. If you need to generate random or example data:
        - Use realistic ranges for the domain
        - Create plausible patterns and relationships
        - Include around 5-10 data points unless specified otherwise

        8. For follow-up questions:
        - Reference previously generated visualizations
        - Reuse previous data with requested modifications
        - Maintain consistent context throughout the conversation

        9. When handling time series, use chronological format for dates/times.

        10. For sports, economic, or technical data where exact values are unknown, always provide representative sample data with a clear explanation that these are approximate/sample values.

        NEVER respond with "I don't have access to real-time data" or similar phrases. Instead, generate plausible sample data and clearly label it as illustrative.
        """
                
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')

        context_system_message = system_message
        if previous_context:
            context_system_message += f"\n\nPrevious visualization context: {json.dumps(previous_context)}"
        
        full_history = [
            {"role": "user", "parts": [{"text": "System instructions: " + context_system_message}]},
            {"role": "model", "parts": [{"text": "I'll help with data visualization following those guidelines."}]},
            *gemini_messages
        ]
        
        chat = model.start_chat(history=full_history)
        
        # Send conversation history and get response
        gemini_response = chat.send_message([{
            "text": f"Using previous context {json.dumps(previous_context) if previous_context else 'None'}, please address: {request.query}"
        }])
        ai_response = gemini_response.text
        
        # Check if response contains plot data
        plot_data = None
        if "plot:" in ai_response:
            plot_spec_start = ai_response.find("plot:")
            plot_spec_end = ai_response.find("\n", plot_spec_start)
            if plot_spec_end == -1:
                plot_spec_end = len(ai_response)
            
            plot_specification = ai_response[plot_spec_start:plot_spec_end]
            plot_image, plot_data = generate_plot(plot_specification)  # Unpack both values
            
            ai_response = ai_response.replace(plot_specification, "")
            
            chat_history.append(Message(
                role="assistant", 
                content=ai_response,
                image=plot_image,
                metadata={"plot_data": plot_data}  # Store plot data in metadata
            ))
            return ChatResponse(response=ai_response, image=plot_image)
        else:
            chat_history.append(Message(
                role="assistant", 
                content=ai_response,
                metadata={"type": "text_only"}
            ))
            return ChatResponse(response=ai_response)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Data Visualization Chatbot API"}