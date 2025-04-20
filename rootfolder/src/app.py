# Import necessary libraries
import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from typing import List, Dict, Any, Optional
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.chains import LLMChain
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
import uvicorn
import asyncio

from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize the LLM with gemini-1.5-flash
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

# Define the research paper content here (you'll replace this with your actual paper)
RESEARCH_PAPER_CONTENT = """
Abstract—With the growth of video content produced by
mobile cameras and surveillance systems, an increasing amount
of data is becoming available and can be used for a variety
of applications such as video surveillance, smart homes, smart
cities, and in-home elder monitoring. Such applications focus
in recognizing human activities in order to perform different
tasks allowing the opportunity to support people in their different
scenarios. In this paper we propose a deep neural architecture for
kitchen human action recognition. This architecture contains an
ensemble of convolutional neural networks connected through
different fusion methods to predict the label of each action.
Experiments show that our architecture achieves the novel stateof-the-art for identifying cooking actions in a well-known kitchen
dataset.
I. INTRODUCTION
Effective assistive applications require accurate identification of the activities that are performed by the user being
helped. Here, activity recognition refers to the task of dealing
with noisy low-level data directly from sensors [1]. Such task
is particularly challenging in the real physical world, since it
either involves fusing information from a number of sensors
or inferring enough information using a single sensor. Failure
to correctly identifying the activity the user is performing has
a cascade effect that often leads to users being frustrated and
giving up using the assistive application.
Single-sensor activity recognition often relies on a video
camera feed [2], which has posed a challenging research
problem in computer vision and machine learning. Advances
in hardware and greater availability of data have allowed
deep learning algorithms, and Convolutional Neural Networks
(CNNs) [3] in particular, to consistently improve on the stateof-the-art. CNNs achieve state-of-the-art results when dealing
with image-based tasks such as object recognition, detection,
and semantic segmentation [4], [5]. Encouraged by those
results, more and more applications are relying on deep neural
architectures to perform video-based tasks [2].
In this paper, we address the problem of recognizing human
activities in an indoor environment with a single static camera.
Our main contribution is on supporting people when they are
in the kitchen, with the final goal of recognizing their actions
when cooking meals. Our approach relies on a deep neural
architecture that comprises multiple convolutional neural networks that are fused prior to performing the action classification. We perform experiments using the Kitchen Scene Context
based Gesture Recognition dataset (KSCGR) [6], and we show
that our proposed approach outperforms the current state-ofthe-art method [7] for this particular dataset.
This paper is organized as follows. Section II details our
novel deep neural architecture for action recognition, whereas
Section III presents a thorough experimental analysis for assessing the performance of our proposed approach. Section IV
points to related work and we finish this paper with our
conclusions and future work directions in Section V.
II. ARCHITECTURE DESIGN
Machine learning algorithms such as artificial neural networks (ANN) have been used to address many challenges of
action and activity recognition. For decades, building machine
learning systems required considerable domain expertise to
create an internal representation (feature construction [8])
from which the learning subsystem could detect or classify
patterns within the input. Deep learning approaches such
as convolutional neural networks mitigate this problem by
automatically learning representations in terms of hierarchical
features, allowing the computer to build complex concepts out
of simpler concepts. In this paper, we develop a deep neural
architecture for action recognition in indoor environments with
a fixed camera using an ensemble of convolutional neural
networks (CNNs). Four different fusion methods including
a support vector machine classifier (SVM) [9] and a long
short-term memory network (LSTM) [10] are used to fuse
the output of the CNNs and provide the final prediction of
the input frame. Our architecture has three main components:
i) data pre-processing, ii) convolutional networks for action
recognition, and iii) fusion strategies for final classification.
Figure 1 illustrates the pipeline of our architecture where
RGB represents the pre-processed dataset with RGB video
frames; OFL represents the pre-processed dataset generated
by dense optical flow; AlexNet, GoogLeNet, and SqueezeNet
are the convolutional neural network architectures we use to
recognize activities; NN is a neural network that weights the
contribution of the probabilities generated by the output of
the previous CNNs; Mean computes the arithmetic mean of
the probabilities provided by the CNNs; SVM is a support
vector machine classifier with linear kernel that classifies the
978-1-5090-6182-2/17/$31.00 ©2017 IEEE 2048
Authorized licensed use limited to: Mukesh Patel School of Technology & Engineering. Downloaded on April 02,2025 at 19:18:13 UTC from IEEE Xplore. Restrictions apply.
Fig. 1. Pipeline of our architecture for action recognition.
probability vectors from the CNNs; and LSTM is a recurrent
neural network architecture that fuses the output probability
vectors from the CNNs to provide the final classification.
We separate the convolutional neural network architectures
into two groups: the pre-trained CNNs and the fully-trained
CNNs. The pre-trained group contains 3 neural networks that
were pre-trained on the ImageNet data [11]. We use the pretrained AlexNet [4], GoogLeNet [12], and SqueezeNet [13]
models freely-available in the Caffe Model Zoo repository1.
The fully-trained group contains a single neural network that
is trained from scratch in the well-known kitchen dataset
KSCGR [6].
The pipeline of our architecture receives images from the
kitchen dataset as input for pre-processing. Pre-processing
extracts dense optical flow representations from the input
images and resizes all images to 256×256, generating two new
input data hereafter called OFL for images with dense optical
flow and RGB for the original RGB data. The system feeds the
pre-trained and fully-trained networks with the RGB and OFL
data, generating output vectors that indicate the probability an
image has of belonging to each class. Each fusion method
(NN, Mean, SVM and LSTM) receives the concatenation of
the probability vectors from the CNNs and predicts the final
class of the input image. In what follows, we further detail
each component of the proposed architecture.
A. Data pre-processing
Pre-processing consists of two steps: image resizing and
optical flow generation. Resizing is important since it reduces
the multidimensional space required by the CNNs to learn
suitable features for image classification, as well as the total
processing time. This step resizes all images of the dataset to
a fixed resolution of 256 × 256. The second step generates
the dense optical flow representation [14] of adjacent frames.
In a nutshell, optical flow represents the 2D displacement of
1https://github.com/BVLC/caffe/wiki/Model-Zoo
pixels between frames generating vectors corresponding to
the movement of points from the first frame to the second.
Dense optical flow generates these displacement vectors, i.e.,
for both horizontal and vertical displacements, regarding all
points within frames. In order to generate the final image
for each sequence of frames, we combine the 2-channel
optical flow vectors and associate color to their magnitude and
direction. Magnitudes are represented by colors and directions
through hue values. The output of the data pre-processing step
consists of two datasets containing the original data with RGB
channels and resized size (RGB), and the optical flow data that
encapsulates motion across frames (OFL).
B. CNN Architectures
In this work, we divided the convolutional neural networks
into two groups: fully-trained and pre-trained networks. The
fully-trained networks have the same architecture and training
hyper-parameters, and they are trained from scratch receiving
the two streams of data (RGB and OFL). The network trained
on RGB is hereafter called GoogLeNet[RGB], whereas the
network trained on OFL is called GoogLeNet[OFL]. Both architectures are 22-layer deep and their inception modules contain
convolutional filters in different scales/resolutions, covering
clusters of diverse information. Each network receives video
frames as input, which traverse several convolutional layers, pooling layers, and fully-connected layers (FC). After a
Softmax layer, the network outputs a vector containing the
probability each frame has of belonging to each class.
Even though a number of off-the-shelf CNN architectures are available [15], [2], in this work we make use
of three pre-trained networks. We choose an architecture
based on inception modules [12] due to its reasonable performance and reduced number of trainable parameters, hereafter
called GoogLeNet[off-the-shelf] and GoogLeNet[Fine-tuned]. The
other two architectures are based on AlexNet [4] (hereafter
called AlexNet[Fine-tuned]) and SqueezeNet [13] (herafter called
SqueezeNet[Fine-tuned]), due to their reduced number of layers and parameters. AlexNet[Fine-tuned], GoogLeNet[off-the-shelf],
GoogLeNet[Fine-tuned], and SqueezeNet[Fine-tuned] were pre
trained on the 1.3-million-image ILSVRC 2012 ImageNet
dataset [11]. Despite the fact that the AlexNet model provided in Caffe Zoo reposity has some small differences from
the original AlexNet by Krizhevsky et al. [4], we do not
believe our results would significantly change due to small
architectural and optimization modifications. Similarly to the
fully-trained networks, after a Softmax layer each network
outputs a vector with the probability of the input image for
each class. The difference between GoogLeNet[off-the-shelf] and
GoogLeNet[Fine-tuned] relies on the fact that in the former we
adjust the last layer to the number of classes of our dataset
and “freeze” the remaining layers during training, i.e., we
do not update weights of any layer but the last. In finetuned networks (AlexNet[Fine-tuned], GoogLeNet[Fine-tuned], and
SqueezeNet[Fine-tuned]), we update all pre-trained layers with
different learning rates, allowing the network to learn features
2049
Authorized licensed use limited to: Mukesh Patel School of Technology & Engineering. Downloaded on April 02,2025 at 19:18:13 UTC from IEEE Xplore. Restrictions apply.
more specific to the target dataset, while starting from a
consistent set of weights.
The idea behind our architecture is that distinct networks
may capture different data patterns. In addition, different views
from the same data may also help in classifying frames into
actions. Thus, the same network processes data with different
representations (RGB and OFL), and three different networks
(AlexNet, GoogLeNet and SqueezeNet) process the same data
(RGB).
C. Fusion Methods
Since the output of each CNN is a vector containing
the probability scores for each class, our model architecture allows for the application of distinct fusion methods for providing the ultimate classification. The fusion
methods intend to merge these vectors in order to increase the accuracy for the action recognition task. Before fusing probabilities, we merge the output of the pretrained networks GoogLeNet[RGB] and GoogLeNet[OFL], generating the GoogLeNet[RGB+OFL] vector. We employ a similar strategy to the fully-trained networks AlexNet[Fine-tuned],
GoogLeNet[Fine-tuned], and SqueezeNet[Fine-tuned], by generating
the 3CNNs vector. The new merged vectors are used as input
to the fusion methods in order to generate predictions for each
class. Figure 1 shows our four different approaches: i) a neural
network (NN) that weights the contribution of the probability
vectors, ii) the standard arithmetic mean, i.e., weight 0.5 for
both vectors (Mean), iii) a multi-class linear Support Vector
Machine (SVM) [9], and iv) a special case of recurrent neural
network called long short-term memory (LSTM) [10].
The NN fusion contains a single-layer neural network to
optimize the weights of the probabilities derived from the
output of the CNNs. Figure 2 illustrates the structure of such
network when using RGB and OFL data, where w1 and w2 are
learned weights, [A] is the vector containing the probabilities
from the output of the CNN that processes the OFL images,
[B] is the vector containing the probabilities for each class
generated by the output of the CNN that processes the RGB
images, and [C] is the vector containing the weighted mean
for each class. The idea behind this neural network is that
its weights (w1 and w2) can be learned automatically by
minimizing a loss function and backpropagating the gradients.
During test time, this fusion method employs the learned
parameters to properly weight the contribution of each merged
vector. The Mean fusion receives the output vector from both
RBG and OFL CNNs and calculates the arithmetic mean for
each class (equal weights), assigning to the image the class
with the highest score. The SVM fusion is based on a multiclass linear Support Vector Machine trained with the output
of the CNNs when using the validation data.At test time, the
SVM predicts the class with the largest score. The LSTM
fusion contains a recurrent neural network in the form of a
chain of repeating modules of weights that intends to learn
long-term dependencies. These long-term dependencies are
represented in the form of previous information connected
to the present image, e.g., the class of the current image is
Fig. 2. Single-layer neural network developed to compute the optimal
weighted average from the outputs of the convolutional neural networks.
represented not only by the information of the current frame,
but also by the information extracted from previous frames.
LSTM units have hidden state augmented with nonlinear
mechanisms to allow states to propagate without modification,
be updated, or be reset using simple learned gating functions
[16].
D. Post processing
Since the process of identifying actions occurs frame by
frame instead of the entire video, sometimes the misclassification of a small number of frames of an action may occur.
Since an activity does not occur in a single frame or in a
very small number of frames, we believe that a frame in the
middle of a sequence of 20 frames that contains a different
class probably suggests that the frame was misclassified. For
example, the misclassification of 5 frames of the Baking action
in the middle of ≈ 200 frames of the None action. Following
the work of Bansal et al. [7], we apply a smoothing process
on the output sequence of classes in order to identify and fix
frames that are probably incorrectly-classified. This smoothing
process consists of sliding a window of fixed-size through
the temporally sorted predicted classes assigning to the target
frame (the frame in the center of the window) the majority
voting of all frames within the window.
"""

# Agent 1: Query Enhancer
class QueryEnhancerAgent:
    def __init__(self):
        self.system_prompt = """You are a Query Enhancement Agent. Your role is to analyze the user's query about a research paper and enhance it to make it more precise and searchable. 
        
        Consider:
        1. Identify key concepts and technical terms in the query
        2. Expand abbreviations if needed
        3. Add relevant related terms that might improve search results
        4. Restructure the query to focus on the most important aspects
        5. Make sure the enhanced query remains faithful to the original intent

        Previous chat history:
        {chat_history}
        
        Provide only the enhanced query without explanation or commentary.
        """
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            HumanMessagePromptTemplate.from_template("{query}")
        ])
        self.chain = self.prompt | llm
    
    def enhance_query(self, query: str, chat_history: Optional[List] = None) -> str:
        invoke_params = {
            "query": query,
            "chat_history": "\n".join([f"{m.type}: {m.content}" for m in chat_history]) if chat_history else ""
        }
        response = self.chain.invoke(invoke_params)
        return response.content
    
class EmbeddingGeneratorAgent:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 

    def generate_paper_embeddings(self, paper_content: str) -> Tuple[FAISS, List[str]]:
        chunks = self.text_splitter.split_text(paper_content)
        vectorstore = FAISS.from_texts(
            texts=chunks,
            embedding=self.embedding_model
        )
        return vectorstore, chunks
    
    def generate_query_embedding(self, query: str):
        return self.embedding_model.embed_query(query)


class DataRetrieverAgent:
    def __init__(self, vectorstore, chunks):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.system_prompt = f"""You are a Data Retrieval Agent specialized in extracting relevant information from research papers.
        
        Your goal is to find the most relevant sections of a research paper that address the user's query.
        Consider the context of the query and focus on retrieving information that directly answers or relates to the question.
        
        The research paper content has been embedded and indexed for semantic search.
        
        Process:
        1. Analyze the semantic meaning of the user's query
        2. Retrieve the most semantically relevant sections from the paper
        3. Focus on accuracy and relevance rather than quantity of information
        
        Return only the most relevant content without additional commentary.
        """
        self.chain = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            HumanMessagePromptTemplate.from_template("User Query: {query}\n\nRetrieved Sections: {sections}")
        ]) | llm
    
    def retrieve_data(self, query: str, k: int = 3) -> str:
        docs = self.vectorstore.similarity_search(query, k=k)       #semantic search on vector store
        retrieved_sections = "\n\n".join([doc.page_content for doc in docs])
        response = self.chain.invoke({
            "query": query, 
            "sections": retrieved_sections
        })
        return response.content, retrieved_sections

class ResponseMergerAgent:
    def __init__(self):
        self.system_prompt = """You are a Response Merger Agent. Your role is to create a comprehensive, accurate response to the user's query by combining retrieved information from a research paper.
        
        Follow these guidelines:
        1. Focus on directly answering the user's query using ONLY the provided retrieved sections
        2. Structure your response logically with a clear introduction, body, and conclusion
        3. Use precise, technical language appropriate for academic content
        4. Include relevant details, findings, and methodologies from the research paper
        5. If the retrieved sections don't contain enough information to answer the query, state this clearly
        6. Do not invent information or add content not present in the retrieved sections
        7. Cite specific parts of the paper when appropriate
        
        Your goal is to provide a response that is both informative and faithful to the original research paper.
        """
        self.chain = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            HumanMessagePromptTemplate.from_template("User Query: {query}\n\nRetrieved Information: {retrieved_info}")
        ]) | llm
    
    def merge_response(self, query: str, retrieved_info: str) -> str:
        response = self.chain.invoke({
            "query": query, 
            "retrieved_info": retrieved_info
        })
        return response.content

# class ResponseValidatorAgent:
#     def __init__(self):
#         self.system_prompt = """You are a Response Validation Agent. Your role is to critically evaluate the response generated for a user query and ensure its accuracy, relevance, and completeness.
        
#         Validation criteria:
#         1. Accuracy: Does the response contain factual errors or misinterpretations of the research paper?
#         2. Relevance: Does the response directly address the user's query?
#         3. Completeness: Does the response cover all necessary aspects of the question?
#         4. Clarity: Is the response clearly written and well-structured?
#         5. Source fidelity: Does the response stay true to the information in the research paper?
        
#         If the response fails any of these criteria, you should return "FAIL: [specific reason]"
#         If the response passes all criteria, you should return "PASS"
#         """
#         self.chain = ChatPromptTemplate.from_messages([
#             SystemMessagePromptTemplate.from_template(self.system_prompt),
#             HumanMessagePromptTemplate.from_template("User Query: {query}\n\nGenerated Response: {response}\n\nRetrieved Information: {retrieved_info}")
#         ]) | llm
    
#     def validate_response(self, query: str, response: str, retrieved_info: str) -> str:
#         validation = self.chain.invoke({
#             "query": query,
#             "response": response,
#             "retrieved_info": retrieved_info
#         })
#         return validation.content

class SourceLabelerAgent:
    def __init__(self):
        self.system_prompt = """You are a Source Labeling Agent. Your role is to identify and label the sources of information used in the response.
        
        For each major point in the response:
        1. Identify which section of the research paper it comes from
        2. Add appropriate source labels to indicate where the information can be found
        3. Ensure the labeling is subtle and doesn't disrupt the flow of the response
        
        Your goal is to make the response more traceable without making it too cluttered with citations.
        """
        self.chain = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            HumanMessagePromptTemplate.from_template("Response: {response}\n\nRetrieved Information: {retrieved_info}")
        ]) | llm
    
    def label_sources(self, response: str, retrieved_info: str) -> str:
        labeled = self.chain.invoke({
            "response": response,
            "retrieved_info": retrieved_info
        })
        return labeled.content

class SummarizerAgent:
    def __init__(self):
        self.system_prompt = """You are a Summarization Agent. Your role is to create concise, accurate summaries of longer content from a research paper.
        
        When summarizing:
        1. Identify and include only the most important points
        2. Preserve the original meaning and key findings
        3. Maintain the technical accuracy of the content
        4. Structure the summary logically
        5. Keep the summary concise while covering all essential information
        
        Your summary should be comprehensive enough to be useful but brief enough to be quickly digested.
        """
        self.chain = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            HumanMessagePromptTemplate.from_template("Content to Summarize: {content}")
        ]) | llm
    
    def summarize(self, content: str) -> str:
        response = self.chain.invoke({"content": content})
        return response.content

class ContradictionDetectorAgent:
    def __init__(self):
        self.system_prompt = """You are a Contradiction Detection Agent. Your role is to identify contradictions between different pieces of information or between different responses.
        
        Focus on:
        1. Logical inconsistencies
        2. Factual contradictions
        3. Conflicting statements or conclusions
        4. Inconsistent methodologies or approaches
        
        If you detect a contradiction, describe it specifically and explain why it's a contradiction.
        If you don't detect any contradictions, simply state "No contradictions detected."
        """
        self.chain = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            HumanMessagePromptTemplate.from_template("Response 1: {response1}\n\nResponse 2: {response2}")
        ]) | llm
    
    def detect_contradictions(self, response1: str, response2: str) -> str:
        contradiction = self.chain.invoke({
            "response1": response1,
            "response2": response2
        })
        return contradiction.content
    
class ConflictResolverAgent:
    def __init__(self):
        self.system_prompt = """You are a Conflict Resolution Agent. Your role is to resolve contradictions or conflicts between different pieces of information from a research paper.
        
        When resolving conflicts:
        1. Identify the specific points of contradiction
        2. Analyze the evidence for each conflicting claim
        3. Determine which position is better supported by the research
        4. Create a coherent explanation that resolves the contradiction
        5. Be transparent about any uncertainty in the resolution
        
        Your goal is to provide a clear, evidence-based resolution to conflicts in the information.
        """
        self.chain = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            HumanMessagePromptTemplate.from_template("Contradicting Information: {contradiction}\n\nOriginal Responses: {response1}\n{response2}")
        ]) | llm
    
    def resolve_conflict(self, contradiction: str, response1: str, response2: str) -> str:
        resolution = self.chain.invoke({
            "contradiction": contradiction,
            "response1": response1,
            "response2": response2
        })
        return resolution.content


class GraphPlannerAgent:
    def __init__(self):
        self.system_prompt = """You are a Graph Planning Agent. Your role is to analyze user queries and determine if and what type of graph would be most appropriate to visualize the information.
        
        When planning visualizations:
        1. Determine if the user query requests or would benefit from a graph
        2. Identify the type of data to be visualized
        3. Select the most appropriate graph type (bar, line, scatter, pie, etc.)
        4. Specify the data points and relationships to be included
        5. Define axes, labels, titles, and other needed components
        
        If no graph is needed or appropriate, clearly state this.
        If a graph is appropriate, provide a structured plan for creating it.
        
        Return your response in JSON format with the following structure:
        {{
            "needs_graph": true/false,
            "graph_type": "bar/line/scatter/pie/etc.",
            "title": "Graph title",
            "x_label": "X-axis label",
            "y_label": "Y-axis label",
            "data_points": [list of key data points needed],
            "additional_notes": "Any additional specifications"
        }}
        """
        self.chain = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            HumanMessagePromptTemplate.from_template("User Query: {query}\n\nRetrieved Information: {retrieved_info}")
        ]) | llm
    
    def plan_graph(self, query: str, retrieved_info: str) -> Dict:
        print("GraphPlanner input:", query, "\n\n", retrieved_info)
        response = self.chain.invoke({
            "query": query,
            "retrieved_info": retrieved_info
        })
        try:
            json_match = re.search(r'({.*})', response.content.replace('\n', ' '), re.DOTALL)
            if json_match:
                plan_json = json.loads(json_match.group(1))
            else:
                plan_json = json.loads(response.content)
            return plan_json
        except json.JSONDecodeError:
            return {"needs_graph": False, "error": "Could not parse graph plan"}

# Agent 11: Graph Generator
class GraphGeneratorAgent:
    def __init__(self):
        self.system_prompt = """You are a Graph Generation Agent. Your role is to extract data from research paper information and create appropriate visualizations based on the graph plan.
        
        When extracting data and generating graphs:
        1. Carefully identify numerical data and relationships mentioned in the text
        2. Structure this data appropriately for visualization
        3. Determine appropriate scales and ranges for axes
        4. Create clear, informative visualizations that accurately represent the data
        5. Include proper titles, labels, and legends
        
        Your response should include:
        1. The extracted data points in a structured format
        2. A Matplotlib script to generate the requested visualization
        
        Return your response in JSON format with the following structure:
        {{
            "extracted_data": {{data structure containing the extracted data}},
            "matplotlib_code": "Python code using matplotlib to generate the graph"
        }}
        """
        self.chain = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            HumanMessagePromptTemplate.from_template("Graph Plan: {graph_plan}\n\nRetrieved Information: {retrieved_info}")
        ]) | llm

    def generate_graph(self, graph_plan: Dict, retrieved_info: str) -> Dict:
        if not graph_plan.get("needs_graph", False):
            return {"graph_generated": False, "reason": "No graph required"}
        
        response = self.chain.invoke({
            "graph_plan": json.dumps(graph_plan), 
            "retrieved_info": retrieved_info
        })
        
        try:
            json_match = re.search(r'({.*})', response.content.replace('\n', ' '), re.DOTALL)
            if json_match:
                generation_data = json.loads(json_match.group(1))
            else:
                generation_data = json.loads(response.content)
                
            # Execute the matplotlib code in a safe environment
            matplotlib_code = generation_data.get("matplotlib_code", "")
            if matplotlib_code:
                # Create a figure and save it to a base64 string
                plt.figure(figsize=(10, 6))
                # Execute the matplotlib code (with safety precautions)
                safe_code = self._sanitize_matplotlib_code(matplotlib_code)
                exec(safe_code)
                
                # Save the plot to a bytes buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
                # Convert to base64 for embedding in HTML
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
                
                return {
                    "graph_generated": True,
                    "image_base64": img_base64,
                    "graph_type": graph_plan.get("graph_type", "unknown")
                }
            else:
                return {"graph_generated": False, "reason": "No matplotlib code provided"}
                
        except Exception as e:
            return {"graph_generated": False, "reason": f"Error generating graph: {str(e)}"}
    
    def _sanitize_matplotlib_code(self, code: str) -> str:
        # Basic sanitization to prevent dangerous code execution
        # In a production environment, you would want more robust sandboxing
        forbidden_imports = ["os", "sys", "subprocess", "eval", "exec"]
        for forbidden in forbidden_imports:
            if f"import {forbidden}" in code or f"from {forbidden}" in code:
                raise ValueError(f"Forbidden import: {forbidden}")
        
        # Only allow matplotlib-related imports
        allowed_imports = ["matplotlib", "numpy", "pandas"]
        import_lines = re.findall(r'import .*|from .* import', code)
        for line in import_lines:
            allowed = False
            for allowed_import in allowed_imports:
                if allowed_import in line:
                    allowed = True
                    break
            if not allowed:
                raise ValueError(f"Forbidden import: {line}")
                
        return code

# Agent 12: Response Generator
class ResponseGeneratorAgent:
    def __init__(self):
        self.system_prompt = """You are a Response Generation Agent. Your role is to create the final response to the user's query, incorporating all validated information and resolving any contradictions.
        
        When generating the final response:
        1. Ensure the response directly addresses the user's query
        2. Incorporate all relevant information from the research paper
        3. Present information in a clear, logical structure
        4. Use appropriate technical language and terminology
        5. Include any source labels provided
        6. Resolve any identified contradictions
        7. Be concise yet comprehensive
        
        Your goal is to provide a high-quality response that accurately represents the information in the research paper.
        """
        self.chain = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            HumanMessagePromptTemplate.from_template("User Query: {query}\n\nValidated Information: {validated_info}\n\nResolved Contradictions: {resolved_contradictions}\n\nSource Labels: {source_labels}\n\nGraph Information: {graph_info}")
        ]) | llm
    
    def generate_response(self, query: str, validated_info: str, resolved_contradictions: str, source_labels: str, graph_info: Dict) -> str:
        response = self.chain.invoke({
            "query": query,
            "validated_info": validated_info,
            "resolved_contradictions": resolved_contradictions if resolved_contradictions else "No contradictions to resolve.",
            "source_labels": source_labels,
            "graph_info": json.dumps(graph_info) if graph_info else "No graph required."
        })
        return response.content

# Agent 13: Memory Storer
class MemoryStorerAgent:
    def __init__(self):
        self.memory = ConversationBufferMemory(return_messages=True,
            memory_key="chat_history",
            output_key="output")
    
    def store_interaction(self, user_query: str, system_response: str):
        self.memory.chat_memory.add_user_message(user_query)
        self.memory.chat_memory.add_ai_message(system_response)
    
    def get_chat_history(self) -> List:
        return self.memory.chat_memory.messages
    
    def clear_memory(self):
        self.memory.clear()

# Agent 14: State Manager
class StateManagerAgent:
    def __init__(self):
        self.current_state = "idle"
        self.state_history = []
        
    def update_state(self, new_state: str):
        self.state_history.append(self.current_state)
        self.current_state = new_state
        
    def get_current_state(self) -> str:
        return self.current_state
        
    def get_state_history(self) -> List[str]:
        return self.state_history

# Agent 15: Error Handler
class ErrorHandlerAgent:
    def __init__(self):
        self.system_prompt = """You are an Error Handling Agent. Your role is to analyze errors that occur during the processing of user queries and provide appropriate responses.
        
        When handling errors:
        1. Identify the type and cause of the error
        2. Determine if the error can be resolved automatically
        3. Provide clear, user-friendly explanations of the error
        4. Suggest possible solutions or next steps
        5. Format the error message in a way that is helpful rather than technical
        
        Your goal is to ensure that even when errors occur, the user receives helpful guidance.
        """
        self.chain = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            HumanMessagePromptTemplate.from_template("Error Type: {error_type}\n\nError Details: {error_details}\n\nUser Query: {query}")
        ]) | llm
    
    def handle_error(self, error_type: str, error_details: str, query: str) -> str:
        response = self.chain.invoke({
            "error_type": error_type,
            "error_details": error_details,
            "query": query
        })
        return response.content

# Main Orchestrator
class MultiAgentOrchestrator:
    def __init__(self):
        try:
        # Generate paper embeddings at initialization
            self.embedding_generator = EmbeddingGeneratorAgent()
            self.vectorstore, self.chunks = self.embedding_generator.generate_paper_embeddings(RESEARCH_PAPER_CONTENT)
            
            # Initialize all agents
            self.query_enhancer = QueryEnhancerAgent()
            self.data_retriever = DataRetrieverAgent(self.vectorstore, self.chunks)
            self.response_merger = ResponseMergerAgent()
            # self.response_validator = ResponseValidatorAgent()
            self.source_labeler = SourceLabelerAgent()
            self.summarizer = SummarizerAgent()
            self.contradiction_detector = ContradictionDetectorAgent()
            self.conflict_resolver = ConflictResolverAgent()
            self.graph_planner = GraphPlannerAgent()
            self.graph_generator = GraphGeneratorAgent()
            self.response_generator = ResponseGeneratorAgent()
            self.memory_storer = MemoryStorerAgent()
            self.state_manager = StateManagerAgent()
            self.error_handler = ErrorHandlerAgent()

        except Exception as e:
            print(f"Initialization error: {e}")
            raise
    
    async def process_query(self, query: str) -> Dict:
        try:
            self.state_manager.update_state("processing_query")
            
            # Step 1: Enhance the query
            enhanced_query = self.query_enhancer.enhance_query(query)
            
            # Step 2: Retrieve relevant data from the research paper
            retrieved_data, raw_retrieved_sections = self.data_retriever.retrieve_data(enhanced_query)
            
            # Step 3: Merge information into a coherent response
            merged_response = self.response_merger.merge_response(enhanced_query, retrieved_data)
            
            # # Step 4: Validate the response
            # validation_result = self.response_validator.validate_response(enhanced_query, merged_response, retrieved_data)
            
            # # Handle failed validation
            # if validation_result.startswith("FAIL"):
            #     self.state_manager.update_state("validation_failed")
            #     # Try to regenerate response
            #     merged_response = self.response_merger.merge_response(enhanced_query, retrieved_data)
            #     validation_result = self.response_validator.validate_response(enhanced_query, merged_response, retrieved_data)
            #     if validation_result.startswith("FAIL"):
            #         return {
            #             "response": f"I'm sorry, but I couldn't generate a reliable answer to your query. The issue was: {validation_result}",
            #             "error": validation_result,
            #             "has_graph": False
            #         }
            
            # Step 5: Label sources
            labeled_response = self.source_labeler.label_sources(merged_response, retrieved_data)
            
            # Step 6: Generate a summary
            summary = self.summarizer.summarize(labeled_response)
            
            # Step 7: Check for contradictions
            contradictions = self.contradiction_detector.detect_contradictions(labeled_response, summary)
            
            # Step 8: Resolve any contradictions
            resolved_content = ""
            if not contradictions.startswith("No contradictions"):
                self.state_manager.update_state("resolving_contradictions")
                resolved_content = self.conflict_resolver.resolve_conflict(contradictions, labeled_response, summary)
            
            # Step 9: Plan any graphs needed
            graph_plan = self.graph_planner.plan_graph(enhanced_query, retrieved_data)
            
            # Step 10: Generate graphs if needed
            graph_data = {}
            if graph_plan.get("needs_graph", False):
                self.state_manager.update_state("generating_graph")
                graph_data = self.graph_generator.generate_graph(graph_plan, retrieved_data)
            
            # Step 11: Generate final response
            self.state_manager.update_state("generating_response")
            final_response = self.response_generator.generate_response(
                enhanced_query, labeled_response, resolved_content, labeled_response, graph_data
            )
            
            # Step 12: Store in memory
            self.memory_storer.store_interaction(query, final_response)
            
            self.state_manager.update_state("completed")
            
            result = {
                "response": final_response,
                "has_graph": graph_data.get("graph_generated", False),
                "graph_data": graph_data if graph_data.get("graph_generated", False) else None
            }
            
            return result
            
        except Exception as e:
            self.state_manager.update_state("error")
            error_message = self.error_handler.handle_error(
                "Processing Error", str(e), query
            )
            return {
                "response": error_message,
                "error": str(e),
                "has_graph": False
            }

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],   # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = MultiAgentOrchestrator()

class QueryRequest(BaseModel):
    query: str

@app.post("/api/query")
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    try:
        result = await orchestrator.process_query(request.query)
        return result
    except Exception as e:
        return {
            "response": f"Oops! Something went wrong while processing your query.",
            "error": str(e),
            "has_graph": False
        }

# WebSocket for real-time updates during processing
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        print("WebSocket connected")  
        
        while True:
            try:
                message = await websocket.receive_text()
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "status": "error",
                        "message": "Invalid JSON data received"
                    })
                    continue
                query = data.get("query", "")
                
                # Send initial status
                await websocket.send_json({"status": "processing"})
                
                # Process query
                result = await orchestrator.process_query(query)
                
                # Send result
                await websocket.send_json({
                    "status": "completed",
                    "result": result
                })
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    "status": "error",
                    "message": "Invalid JSON data received"
                })
            except Exception as e:
                await websocket.send_json({
                    "status": "error",
                    "message": str(e)
                })
                
    except WebSocketDisconnect:
        print("WebSocket disconnected")  # Debug log
    except Exception as e:
        print(f"WebSocket error: {e}")  # Debug log


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)