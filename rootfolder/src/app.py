# Import necessary libraries
import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from typing import List, Dict, Any, Optional, Tuple
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
Deep Neural Networks for
Kitchen Activity Recognition
Juarez Monteiro, Roger Granada, Rodrigo C. Barros, and Felipe Meneguzzi
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
III. EXPERIMENTAL ANALYSIS
In this section, we describe the dataset used in our experiments for indoor fixed-camera action recognition, the implementation details regarding the CNNs and fusion methods, and
the results that were achieved by our approach in comparison
with the current state-of-the-art.
A. KSCGR Dataset
The Kitchen Scene Context based Gesture Recognition
dataset2 (KSCGR)[6] is a fine-grained kitchen action dataset
released as a challenge in ICPR 20123. The dataset contains
scenes captured by a kinect sensor fixed on the top of the
kitchen, providing synchronized color and depth image sequences. Each video is 5 to 10 minutes long, containing 9,000
2http://www.murase.m.is.nagoya-u.ac.jp/KSCGR/
3http://www.icpr2012.org/
2050
Authorized licensed use limited to: Mukesh Patel School of Technology & Engineering. Downloaded on April 02,2025 at 19:18:13 UTC from IEEE Xplore. Restrictions apply.
Fig. 3. Example of the frame/action sequence for the “ham and egg” menu.
to 18,000 frames. The organizers of the dataset assigned labels
to each frame indicating the type of gesture performed by the
actors. There are 8 cooking gestures in the dataset: breaking,
mixing, baking, turning, cutting, boiling, seasoning, peeling,
and none, where none means that there is no action being
performed in the current frame. These gestures are performed
in five different menus for cooking eggs in Japan: ham and
eggs, omelet, scrambled egg, boiled egg, and kinshi-tamago.
A total of 7 different subjects perform each menu. The ground
truth data contains the frame id and the action being performed
within the frame.
We divided the dataset into training, validation, and test sets.
The training set contains 4 subjects, each of them performing
5 recipes, i.e., 20 videos and 139,196 frames in total. We
use the validation set to obtain the model configuration that
performs best, i.e., the configuration with the highest accuracy.
This set contains 1 subject performing 5 recipes with 32,897
frames in total. We use the test set to assess the accuracy
of the selected model in unseen data. This set contains 2
subjects, each performing 5 recipes, i.e., 10 videos with 55,781
frames in total. Figure 3 shows the sequence of frames and
actions when performing the menu Ham and Egg, where the
colored bar represents the timeline of appearance of frames
and actions, and the images illustrate examples of each action
performed in the video.
B. Implementation
Fully-trained CNNs architecture: in order to perform the
action recognition task we use an inception-based CNN architecture [12] trained from scratch in RGB and OFL separately.
The training phase uses mini-batch stochastic gradient with
momentum (0.9). For each iteration, the network forwards a
mini-batch of 128 samples. We apply data augmentation with
random crops, i.e., a different crop in a randomly selected part
of the image is selected, as well as a probabilistic horizontal
flip, generating a sub-image of 224 × 224. All images have
their pixels subtracted by the mean pixel values of all training
images. All convolutions, including those within the inception
modules, use rectified linear activation units (ReLU). Regarding weight initialization, we employ the Xavier algorithm that
automatically determines the value of initialization based on
the number of input neurons. To minimize the chances of
overfitting, we apply dropout on the fully-connected layers
with a probability of 70%. The learning rate is set to 10−3
and we drop it by a factor of 50 every epoch, stopping the
training after 43.5k iterations (30 epochs).
Pre-trained CNNs architectures: all networks of this
group were pre-trained over the ILSVRC 2012 ImageNet
dataset [11]. For the training phase, we kept almost the same
configuration for all networks, using a mini-batch of 128 samples with a random crop of 224 × 224 as well as random horizontal flip. Each image has its pixels subtracted by the mean
value of pixels of each channel. During training, we freeze
all but the last layer of GoogLeNet[off-the-shelf], performing the
weights and bias updates only for the last fully-connected layer
for 10 epochs, increasing the learning rate of the layer by 10
(setting learning rate of the weights to 10 and learning rate
of the bias to 20). For fine-tuned models (AlexNet[Fine-tuned],
GoogLeNet[Fine-tuned], and SqueezeNet[Fine-tuned]), we update all
weights but with a different learning rate for the last layer. We
increase the learning rate of the weights in the last layer from
1 to 10 and the bias from 2 to 20, and decrease the global
learning rate by 100. This configuration allows all layers to
learn, though giving the final layer the capability to learn faster
than the remaining layers.
NN: this fusion approach contains a neural network trained
with data from the validation set for 10 epochs with weights
w1 and w2 initialized with 0.5. We use the mean squared
error loss function and optimize it through Adam [17] with
a learning rate set to 10−3.
SVM: we train the multi-class Support Vector Machine
using the off-the-shelf implementation by Crammer and Singer
[9] from scikit-learn4 toolbox. Similarly to the neural network
fusion, we train the SVM using the validation set. We use the
linear kernel and default scikit-learn regularization parameter
C = 1 with the square of the hinge loss as loss function.
LSTM: we implemented the long short-term memory using
the Keras5 neural networks library. Our configuration follows
the implementation proposed by Donahue et al. [16] that
connects a CNN with a LSTM, calling this model Longterm Recurrent Convolutional Network (LRCN). We explore
various hyper-parameters using both training and validation
sets, selecting the best architecture that contains 1024 hidden
units with a dropout of 0.7 in order to avoid overfitting. We
train and test the LSTM network in a sequence of 32 frames,
and during training the stride is of 16 frames. We also apply
the Adam [17] algorithm using a learning rate of 10−3. We
run the training phase for 30 epochs.
Post processing: the post processing consists of sliding a
window of fixed-size through the predicted classes assigning
to the target frame the majority voting of all frames within the
window. In order to decide the size of the window, we used the
predicted classes from the validation dataset. We performed
several smoothing tests, varying the window-size from 10 to
50 increasing the step in 10 frames each time. Finally, we
4http://scikit-learn.org
5https://keras.io
2051
Authorized licensed use limited to: Mukesh Patel School of Technology & Engineering. Downloaded on April 02,2025 at 19:18:13 UTC from IEEE Xplore. Restrictions apply.
TABLE I
PER-ACTIVITY ACCURACY IN THE KSCGR DATASET FOR ALL BASELINES AND FUSION METHODS.
Method None Breaking Mixing Baking Turning Cutting Boiling Seasoning Peeling Overall
GoogLeNet[RGB] 0.644 0.275 0.289 0.671 0.346 0.588 0.287 0.363 0.117 0.689
GoogLeNet[OFL] 0.519 0.341 0.314 0.600 0.194 0.545 0.128 0.382 0.449 0.631
GoogLeNet[RGB+OFL] + Mean 0.634 0.327 0.340 0.684 0.174 0.620 0.169 0.403 0.347 0.692
GoogLeNet[RGB+OFL] + SVM 0.679 0.357 0.432 0.689 0.000 0.526 0.444 0.601 0.455 0.721
GoogLeNet[RGB+OFL] + NN 0.690 0.354 0.452 0.693 0.012 0.516 0.505 0.651 0.382 0.726
GoogLeNet[Off-the-shelf] 0.545 0.004 0.198 0.666 0.009 0.182 0.340 0.055 0.007 0.609
AlexNet[Fine-tuned] 0.688 0.555 0.445 0.752 0.211 0.636 0.369 0.661 0.400 0.751
GoogLeNet[Fine-tuned] 0.579 0.224 0.374 0.711 0.136 0.438 0.030 0.174 0.000 0.645
SqueezeNet[Fine-tuned] 0.611 0.325 0.422 0.688 0.117 0.313 0.078 0.300 0.184 0.660
AlexNet[Fine-tuned] + SVM 0.636 0.570 0.395 0.741 0.173 0.447 0.303 0.520 0.335 0.717
GoogLeNet[Fine-tuned] + SVM 0.676 0.323 0.466 0.708 0.100 0.449 0.381 0.351 0.202 0.712
SqueezeNet[Fine-tuned] + SVM 0.538 0.211 0.240 0.593 0.028 0.010 0.078 0.113 0.013 0.587
3CNNs + SVM 0.604 0.363 0.449 0.678 0.105 0.269 0.165 0.236 0.042 0.667
3CNNs + NN 0.687 0.598 0.434 0.757 0.209 0.623 0.348 0.663 0.509 0.752
3CNNs + NN + PP 0.696 0.621 0.452 0.753 0.206 0.575 0.333 0.725 0.509 0.755
3CNNs + LSTM 0.737 0.508 0.536 0.739 0.191 0.571 0.458 0.416 0.738 0.775
3CNNs + LSTM + PP 0.754 0.504 0.564 0.749 0.190 0.560 0.469 0.384 0.773 0.785
chose the window size of 20 frames since it achieved the best
accuracy results on validation data.
C. Results
In order to evaluate our approach, we compare the output
of each fusion method in the test set. We use the classification
generated by each individual CNN as baseline, and hence
we can see whether the fusion method improves over each
individual CNN. Table I shows the accuracy values for each
class individually (None, Breaking, Mixing, Baking, Turning,
Cutting, Boiling, Seasoning, Peeling), as well as the overall
accuracy (Overall) that considers all classes at once.
We generate values of accuracy for the fullytrained models: GoogLeNet trained with RGB
(GoogLeNet[RGB]) and OFL (GoogLeNet[OFL]) data, a
merging of both networks GoogLeNet[RGB+OFL] with
either the Mean (GoogLeNet[RGB+OFL]+Mean), the SVM
(GoogLeNet[RGB+OFL]+SVM) or the neural network
(GoogLeNet[RGB+OFL]+NN) as the fusion method. For
pre-trained models, we generate values of accuracy for an
off-the-shelf network (GoogLeNet[Off-the-shelf]), and for finetuned networks (AlexNet[Fine-tuned], GoogLeNet[Fine-tuned], and
SqueezeNet[Fine-tuned]). We test the fine-tuned models changing
the output to a support vector machine classifier, generating
AlexNet[Fine-tuned]+SVM, GoogLeNet[Fine-tuned]+SVM and
SqueezeNet[Fine-tuned]+SVM. The merging of the pre-trained
networks (3CNNs) is also used as input to the fusion methods
generating 3CNNs+SVM, 3CNNs+NN, and 3CNNs+LSTM.
Finally the post processing (PP) is applied on the output
of the neural network generating 3CNNs+NN+PP and
3CNNs+LSTM+PP.
1) Overall Performance: As we can observe in Table I,
the fusion of 3 pre-trained convolutional neural networks
with a long short-term memory network and with the post
processing strategy (3CNNs+LSTM+PP) achieves the best
global accuracy (All) of 78.5%. The achieved results confirm
our belief that different networks may identify different aspects
(features) and their combination tends to improve results, as
largely expected due to the ensemble effect. When comparing
the merging of the 3 networks with single networks, 3CNN
achieves the best results for 6 (None, Breaking, Mixing, Baking, Seasoning, and Peeling) out of 9 actions. For Cutting and
Boiling, our architecture using 3 networks achieves the secondbest result. Our architecture did not perform well for the
Turning action, and a possible reason for the low performance
of the fusion methods for classifying Turning might be a
mixture of the limited number of frames for this activity and
the training phase using vector probabilities generated based
on the validation set. Considering that our fusion methods are
trained with predicted probabilities from validation data, any
misclassification may lead to errors during the test phase.
2) Off-the-shelf vs. fully-trained vs. fine-tuned: In general,
the GoogLeNet[Off-the-shelf] architecture is outperformed by its
fully-trained version on the RGB data GoogLeNet[RGB] and
by its fine-tuned version GoogLeNet[Fine-tuned]. This result
indicates that it is better to train the network from scratch
when a large dataset is available or fine-tune the network
allowing all layers to learn instead of simply learning the last
layer. Comparing the network trained from scratch with the
fine-tuned network, it seems better to train the network from
scratch than to load pre-trained weights in ImageNet. These
results may be explained by the fact that KSCGR’s images are
very different from ImageNet’s.
3) Single vs. Merged vs. Merged/Fused : The merging
of networks using different datasets (GoogLeNet[RGB+OFL])
with a fusion method (Mean, SVM or NN) tends to
improve results, achieving the maximum accuracy of
≈ 73% when combining the network with a neural network fusion (GoogLeNet[RGB+OFL]+NN). The use of
trainable fusion methods (GoogLeNet[RGB+OFL]+SVM and
GoogLeNet[RGB+OFL]+NN) decrease the accuracy of the Turning action probably because the validation data contains very
few frames from this action. When comparing the finetuned networks with their versions with the SVM fusion,
we can see that the original fine-tuned network achieves
better results. GoogLeNet[Fine-tuned] + SVM is the only network
2052
Authorized licensed use limited to: Mukesh Patel School of Technology & Engineering. Downloaded on April 02,2025 at 19:18:13 UTC from IEEE Xplore. Restrictions apply.
that achieves better results (7 out of 9 classes) when using
a fusion algorithm. Virtually every result of the original
fine-tuned versions of AlexNet and SqueezeNet are better
than their versions using SVM. Comparing the three finetuned networks (AlexNet[Fine-tuned], GoogLeNet[Fine-tuned] and
SqueezeNet[Fine-tuned]), AlexNet[Fine-tuned] achieves the best results for 7 out of 9 classes and the best overall class score.
These results indicate that a small network is capable of good
performance probably because they properly avoid overfitting.
Observing the 3 fusion methods, the SVM fusion achieves
the worst results for most cases. NN and LSTM obtain similar results for most categories, and 3CNNs+LSTM achieves
the best overall classification with 77% of accuracy for all
classes. Even though post processing may eliminate classes
that contain a small number of frames, it seems its usage
is quite beneficial since it consistently improves the obtained
results. The post-processed versions achieve the best accuracy
scores for 5 out of 9 classes when compared with all models,
providing the best overall accuracy of 78.5%.
4) Unbalanced classes: Since classification accuracy takes
into account only the proportion of correct results that a
classifier achieves, it is not suitable for unbalanced datasets
because it is biased towards classes with larger number of
examples. Although other factors may change results, classes
with a larger number of examples tend to achieve better results
since the network has more examples to learn the variability
of the features. By analyzing the KSCGR dataset, we note that
it is indeed unbalanced, i.e., classes are not equally distributed
over frames. Figure 4 shows the distribution of accuracy scores
over the classes for the GoogLeNet[RGB] network (left) and
the distribution of these classes within the dataset. We can
see that the dataset is unbalanced since the None action has
the largest number of frames (≈ 30% of the total) followed
by Baking (≈ 25% of the total), whereas Breaking contains
only ≈ 3% of the frames. By checking the accuracy scores,
we see that the GoogLeNet[RGB] achieves 28% of accuracy for
the Breaking class and 67% of accuracy for the Baking class,
meaning that the features that map to Breaking are not as
evident as the features of Baking. The probable reason for this
difference relies on the small number of training examples of
the Breaking class that is passed to the network. Baking, on the
other hand, is much more present within the dataset, improving
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy of the GoogLeNet[RGB] network
3% 31%
11%
25%
5% 9%
7%
3%
6%
Distribution of frames in dataset
None
Breaking
Mixing
Baking
Turning
Cutting
Boiling
Seasoning
Peeling
Fig. 4. Per-class accuracy and class distribution within the KSCGR dataset.
the training experience and making the neural architecture
generalize better for frames that belong to that class. For
dealing with the unbalanced nature of the KSCGR dataset,
we measure the performance of the fusion methods based
on precision (P), recall (R), and F-Measure (F). Table II
shows the values of precision, recall, F-measure, and accuracy
achieved by the baselines (GoogLeNet [RGB] and GoogLeNet
[OFL]), pre-trained, and fully trained networks. In order
to compare with the current state-of-the-art on the KSCGR
dataset, in Table II we present the performance achieved by
Bansal et al. [7], which uses hand-crafted features (HCF) to
identify activities, as well as their results after undergoing
a similar post-processing step (HCF+PP). A large precision
value means that the respective model can adjust very well
to the features for identifying the class, whereas low values
indicate that it cannot extract relevant features to identify the
correct class among the remaining classes.
5) Our approach vs State-of-the-art: As we can observe
in Table II, virtually all networks achieve similar values of
precision, recall, and F-Measure. It is important to note that
the combination of the 3 CNNs using the LSTM as fusion
method and post processing (3CNNs+LSTM+PP) achieves the
best scores for all measures. When compared with the handcrafted features proposed by Bansal et al. [7], it is clear that
our architecture provides better results. Only the network that
was pre-trained on ImageNet and had all but the last layer
frozen (GoogLeNet[Off-the-shelf]) is outperformed by the (former)
state-of-the-art.
6) Confusion Matrix: We also analyze the confusion matrix
of the network that achieves the best performance without
post processing since it is also important to see which classes
that are commonly mistaken. The normalized confusion matrix depicted in Figure 5 shows the performance of the
3CNNs+LSTM network, where rows represent the true classes
TABLE II
PRECISION, RECALL, F-MEASURE, AND ACCURACY FOR ALL BASELINES,
FUSION METHODS AND THE (FORMER) STATE-OF-THE-ART APPROACH
FOR THE KSCGR DATASET.
Approach Precision Recall F-measure Accuracy
HCF [7] 0.62 0.63 0.61 0.64
HCF + PP [7] 0.68 0.68 0.68 0.72
GoogLeNet[RGB] 0.69 0.68 0.69 0.69
GoogLeNet[OFL] 0.64 0.63 0.63 0.63
GoogLeNet[Off-the-shelf] 0.61 0.61 0.61 0.61
GoogLeNet[RGB+OFL] + Mean 0.71 0.69 0.70 0.69
GoogLeNet[RGB+OFL] + SVM 0.67 0.72 0.70 0.72
GoogLeNet[RGB+OFL] + NN 0.72 0.73 0.72 0.73
AlexNet[Fine-tuned] 0.77 0.75 0.76 0.75
GoogLeNet[Fine-tuned] 0.73 0.71 0.72 0.71
SqueezeNet[Fine-tuned] 0.70 0.66 0.68 0.66
AlexNet[Fine-tuned] + SVM 0.76 0.72 0.74 0.72
GoogLeNet[Fine-tuned] + SVM 0.72 0.71 0.68 0.71
SqueezeNet[Fine-tuned] + SVM 0.64 0.59 0.61 0.59
3 CNNs + SVM 0.73 0.67 0.70 0.67
3 CNNs + NN 0.77 0.75 0.76 0.75
3 CNNs + NN + PP 0.77 0.76 0.75 0.76
3 CNNs + LSTM + PP 0.80 0.78 0.79 0.79
3 CNNs + LSTM 0.78 0.78 0.78 0.78

None
Breaking
Mixing
Baking
Turning
Cutting
Boiling
Seasoning
Peeling
Predicted label
None
Breaking
Mixing
Baking
Turning
Cutting
Boiling
Seasoning
Peeling
True label
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
Fig. 5. Normalized confusion matrix for the 3CNNs+LSTM network.
and columns the predicted classes. Shades of blue represent
the value in each cell, going chromatically from a darker
blue for higher values to a lighter blue for lower values.
The confusion matrix shows normalized values, i.e., predicted
values are divided by the total number of true values for each
cell. By analyzing the results for the Baking class, we can
see that the system incorrectly predicts it as Turning. Such
misclassification makes sense since both activities occur in the
same region of the frame, using the same objects (e.g., in both
activities the subject is working in the middle of the scene and
whereas in Baking the subject puts the broken egg onto the
pan and lets it bake, in Turning the subject turns the baked egg
on the pan. Even though both None, Mixing, and Baking have
higher values in the main diagonal of the confusion matrix,
their precision scores are reduced by the misclassification of
other classes. Similarly to the misclassification of Baking, the
Mixing and Turning activities occur in the same region of the
scene and with the same objects.
Unlike Baking that does not have many changes through
frames (e.g., the egg baking inside the pan), the None activity
is labeled as anything that happens but the other 8 activities,
encompassing frames in which the subject is preparing the
kitchen utensils, moving pans, and inter-activity frames such as
removing the egg from boiling to peeling. The large accuracy
(73%) for classifying Baking may be explained due to this
standard behavior of low-variability in regions of the scene
and the larger number of available training frames. Despite the
unbalanced nature of the dataset, the values of accuracy follow
the F-Measure scores, where the lowest value is achieved for
Turning and the largest value for Baking.
7) Post processing effect: Since the post processing strategy
seems to be successful, it is interesting to visualize the exact
effect of smoothing predictions. Figure 6 presents the temporal
representation of the class distribution in the frame sequence
for a single video of the test set. Classes are represented by
Fig. 6. Temporal representation of classes for the frame sequence of a single
video in which true labels are compared with labels predicted by our approach
(with and without post processing.
colored vertical lines in a temporal sequence for both the original output (true labels), the output provided by 3CNNs+LSTM,
and the output provided by (3CNNs+LSTM+PP). By analyzing the output of 3CNNs+LSTM, we can see that some
frames are misclassified such as a single Mixing action in
the middle of the Baking class or a small sequence of the
Cutting action in the middle of a Seasoning action. After
performing the post-processing smoothing step, these noisy
predictions disappear. On the other hand, small sequence of
frames that were correctly classified (Mixing in the middle of a
Baking class) also disappear. Despite the increase in accuracy, the
smoothed output completely ignored the existence of some
activities, which can be an important issue according to the
application at hand.
IV. RELATED WORK
Before the rise of CNN and neural networks in general,
the approaches for action recognition were based on complex
hand-crafted features extracted from video sequences [7]. Convolutional neural networks on the other hand, learn automatically a hierarchy of features automating the process of feature
construction. Thus, many authors apply CNNs to recognize
actions in videos using methods such as Long-term Recurrent
Convolutional Network (LRCN) [16], 3D convolutions (3D
CNNs) [18], or a mix of hand-crafted features and CNNs (twostream CNNs) [15]. Despite the fact that these approaches are
suitable for recognizing activities in general, they were applied
in other datasets and are not directly comparable to our work.
Traditional approaches for activity recognition rely on handcrafted features and domain-specific image processing algorithms and often result in limited accuracy [7], [19]. Bansal et
al. [7] perform daily life cooking activity recognition based
on hand-crafted features for hand movements and objects
use in KSCGR dataset [6]. Their method first detects hand
regions through color segmentation and skin identification.
Since some objects can have the same color of the skin.
"""

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


class ContextManagerAgent:
    def __init__(self):
        self.system_prompt = """You are a Context Management Agent. Your role is to analyze the user's enhanced query and previous chat history to establish relevant context for the current query.
        
        Consider:
        1. Identify how the current query relates to previous interactions
        2. Determine what context from previous exchanges is relevant to the current query
        3. Detect shifts in topic or focus that require new context
        4. Extract any constraints or preferences mentioned in the conversation history
        5. Formulate a concise context summary to guide the information retrieval process
        
        Previous chat history:
        {chat_history}
        
        Enhanced query:
        {enhanced_query}
        
        Provide a concise context summary that will help guide the information retrieval process.
        """
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            HumanMessagePromptTemplate.from_template("Enhanced Query: {enhanced_query}")
        ])
        self.chain = self.prompt | llm
    
    def establish_context(self, enhanced_query: str, chat_history: Optional[List] = None) -> str:
        invoke_params = {
            "enhanced_query": enhanced_query,
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
            HumanMessagePromptTemplate.from_template("User Query: {query}\n\nQuery Context: {context}\n\nRetrieved Sections: {sections}")
        ]) | llm
    
    def retrieve_data(self, query: str, context: str, k: int = 3) -> str:
        docs = self.vectorstore.similarity_search(query, k=k)       #semantic search on vector store
        retrieved_sections = "\n\n".join([doc.page_content for doc in docs])
        response = self.chain.invoke({
            "query": query, 
            "context": context,
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
            HumanMessagePromptTemplate.from_template("User Query: {query}\n\nQuery Context: {context}\n\nRetrieved Information: {retrieved_info}")
        ]) | llm
    
    def merge_response(self, query: str, context: str, retrieved_info: str) -> str:
        response = self.chain.invoke({
            "query": query,
            "context": context,
            "retrieved_info": retrieved_info
        })
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
            HumanMessagePromptTemplate.from_template("Merged Response: {merged_response}\n\nRetrieved Information: {retrieved_info}")
        ]) | llm
    
    def detect_contradictions(self, merged_response: str, retrieved_info: str) -> str:
        contradiction = self.chain.invoke({
            "merged_response": merged_response,
            "retrieved_info": retrieved_info
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
            HumanMessagePromptTemplate.from_template("Contradicting Information: {contradiction}\n\nMerged Response: {merged_response}\n\nRetrieved Information: {retrieved_info}")
        ]) | llm
    
    def resolve_conflict(self, contradiction: str, merged_response: str, retrieved_info: str) -> str:
        if contradiction.startswith("No contradictions"):
            return ""
        
        resolution = self.chain.invoke({
            "contradiction": contradiction,
            "merged_response": merged_response,
            "retrieved_info": retrieved_info
        })
        return resolution.content


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
            HumanMessagePromptTemplate.from_template("User Query: {query}\n\nQuery Context: {context}\n\nResponse Content: {response_content}\n\nRetrieved Information: {retrieved_info}")
        ]) | llm
    
    def plan_graph(self, query: str, context: str, response_content: str, retrieved_info: str) -> Dict:
        print("GraphPlanner input:", query, "\n\n", retrieved_info)
        response = self.chain.invoke({
            "query": query,
            "context": context,
            "response_content": response_content,
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
                
            matplotlib_code = generation_data.get("matplotlib_code", "")
            if matplotlib_code:
                plt.figure(figsize=(10, 6))
                safe_code = self._sanitize_matplotlib_code(matplotlib_code)
                exec(safe_code)
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
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
        forbidden_imports = ["os", "sys", "subprocess", "eval", "exec"]
        for forbidden in forbidden_imports:
            if f"import {forbidden}" in code or f"from {forbidden}" in code:
                raise ValueError(f"Forbidden import: {forbidden}")
        
        code = code.replace("plt.show()", "")

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

# Agent 13: Response Generator
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
            HumanMessagePromptTemplate.from_template("User Query: {query}\n\nQuery Context: {context}\n\nLabeled Content: {labeled_content}\n\nSummary: {summary}\n\nResolved Contradictions: {resolved_contradictions}\n\nGraph Information: {graph_info}")
        ]) | llm
    
    def generate_response(self, query: str, context: str, labeled_content: str, summary: str, resolved_contradictions: str, graph_info: Dict) -> str:
        response = self.chain.invoke({
            "query": query,
            "context": context,
            "labeled_content": labeled_content,
            "summary": summary,
            "resolved_contradictions": resolved_contradictions if resolved_contradictions else "No contradictions to resolve.",
            "graph_info": json.dumps(graph_info) if graph_info else "No graph required."
        })
        return response.content

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

class MultiAgentOrchestrator:
    def __init__(self):
        try:
            self.state_manager = StateManagerAgent()
            self.query_enhancer = QueryEnhancerAgent()
            self.context_manager = ContextManagerAgent()
            self.embedding_generator = EmbeddingGeneratorAgent()
            self.vectorstore, self.chunks = self.embedding_generator.generate_paper_embeddings(RESEARCH_PAPER_CONTENT)
            self.data_retriever = DataRetrieverAgent(self.vectorstore, self.chunks)
            self.response_merger = ResponseMergerAgent()
            self.contradiction_detector = ContradictionDetectorAgent()
            self.conflict_resolver = ConflictResolverAgent()
            self.source_labeler = SourceLabelerAgent()
            self.summarizer = SummarizerAgent()
            self.graph_planner = GraphPlannerAgent()
            self.graph_generator = GraphGeneratorAgent()
            self.response_generator = ResponseGeneratorAgent()
            self.memory_storer = MemoryStorerAgent()
            self.error_handler = ErrorHandlerAgent()

        except Exception as e:
            print(f"Initialization error: {e}")
            raise
    
    async def process_query(self, query: str) -> Dict:
        try:
            self.state_manager.update_state("processing_query")
            
            self.state_manager.update_state("enhancing_query")
            enhanced_query = self.query_enhancer.enhance_query(query, self.memory_storer.get_chat_history())
            
            self.state_manager.update_state("establishing_context")
            context = self.context_manager.establish_context(enhanced_query, self.memory_storer.get_chat_history())
            
            self.state_manager.update_state("retrieving_data")
            
            retrieved_data, raw_retrieved_sections = self.data_retriever.retrieve_data(enhanced_query, context)
            
            self.state_manager.update_state("merging_response")
            merged_response = self.response_merger.merge_response(enhanced_query, context, retrieved_data)
            
            self.state_manager.update_state("detecting_contradictions")
            contradictions = self.contradiction_detector.detect_contradictions(merged_response, retrieved_data)
            
            self.state_manager.update_state("resolving_contradictions")
            resolved_content = self.conflict_resolver.resolve_conflict(contradictions, merged_response, retrieved_data)
            
            self.state_manager.update_state("labeling_sources")
            labeled_response = self.source_labeler.label_sources(merged_response, retrieved_data)
            
            self.state_manager.update_state("summarizing")
            summary = self.summarizer.summarize(labeled_response)
            
            self.state_manager.update_state("planning_graph")
            graph_plan = self.graph_planner.plan_graph(enhanced_query, context, labeled_response, retrieved_data)
            
            graph_data = {}
            if graph_plan.get("needs_graph", False):
                self.state_manager.update_state("generating_graph")
                graph_data = self.graph_generator.generate_graph(graph_plan, retrieved_data)

            self.state_manager.update_state("generating_response")
            final_response = self.response_generator.generate_response(
                enhanced_query, context, labeled_response, summary, resolved_content, graph_data
            )
            
            self.state_manager.update_state("storing_memory")
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
    allow_origins=["http://localhost:3000"],   
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
        print("WebSocket disconnected") 
    except Exception as e:
        print(f"WebSocket error: {e}")  


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
