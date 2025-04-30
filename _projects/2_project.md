---
layout: page
title: Gaze-Guided Reinforcement Learning for Visual Search
description: Discover how gaze prediction from human eye-tracking enhances AI agents in object search tasks. By integrating visual attention into reinforcement learning through three novel methods, our approach enables faster, more effective navigation in simulated environments.
img: assets/img/proj2-1.jpg
importance: 1
category: Work
related_publications: false
---

This blog explores how we can make AI agents search for objects more efficiently by mimicking human visual attention patterns. Using a gaze prediction model trained on human eye-tracking data, we've developed three innovative approaches to integrate this visual attention information into a reinforcement learning framework: channel integration (adding gaze as a fourth input channel), bottleneck integration (processing RGB and gaze separately before combining), and weighted integration (using gaze to modulate visual inputs).

Our experiments in simulated indoor environments demonstrate that these gaze-guided agents learn more efficiently and navigate more effectively than traditional approaches. By bridging cognitive science and machine learning, this project offers insights into how biologically-inspired attention mechanisms can enhance AI systems for robotics, assistive technologies, and autonomous navigation.

---

#### **Introduction and Motivation**

Visual search is a fundamental task that humans perform effortlessly every day. Whether looking for your keys, finding a product on a shelf, or locating a friend in a crowd, our visual system efficiently guides our attention to relevant objects. However, autonomous agents struggle with these same tasks, often employing inefficient search strategies that waste time and computational resources.

This project explores a novel approach: leveraging human gaze patterns to enhance reinforcement learning for visual search tasks. By integrating human attention data, we aim to teach AI agents to "look where humans look," dramatically improving search efficiency in environments like AI2-THOR, a realistic 3D household simulator.

The motivation behind this research stems from a simple observation: humans use sophisticated attention mechanisms developed through evolution to efficiently prioritize visual information. When we search for objects, our eyes don't scan uniformly across a scene; instead, we rapidly focus on relevant locations based on context, object relationships, and prior knowledge. For instance, when looking for a microwave in a kitchen, humans instinctively focus on countertops and walls rather than floors or ceilings.

This gaze-guided approach has significant practical applications:

- **Household robots** that can efficiently find and manipulate objects
- **Assistive technology** for visually impaired individuals that can locate objects upon request
- **Search and rescue robots** that can quickly identify people or objects in disaster scenarios
- **Warehouse automation** systems that locate specific products among thousands

The fundamental research question we explore is: Can we leverage human gaze patterns to improve reinforcement learning efficiency in object search tasks? Our hypothesis is that by incorporating human visual attention data as an additional signal in the reinforcement learning process, we can achieve faster learning, better generalization, and more human-like search behavior.

In this project, we implement multiple methods for integrating gaze information with visual data, train agents to search for objects, and evaluate their performance across different scenarios. The results demonstrate that gaze-guided reinforcement learning significantly outperforms traditional approaches, opening new possibilities for AI systems that can see the world more like we do.

---

#### **Background and Context**

**Visual Search in AI** 

Visual search in AI presents numerous challenges that humans solve intuitively. Traditional AI approaches often rely on exhaustive exploration strategies that lack the efficiency of human visual search. In environments like AI2-THOR, as implemented in our project, agents must process high-dimensional visual inputs, maintain spatial awareness, and make decisions with limited information—all while dealing with partial observability where only a fraction of the environment is visible at any time. These challenges make autonomous visual search computationally expensive and time-consuming compared to human performance.

**Human Attentional Mechanisms** 

Humans excel at visual search through sophisticated attentional mechanisms. Our visual system employs both bottom-up attention (driven by salient features like color and motion) and top-down attention (guided by goals and prior knowledge). When searching for objects, we don't scan uniformly across scenes but rather prioritize locations where target objects are likely to appear. For example, when searching for a microwave in a kitchen, we instinctively focus on countertops and cabinet areas while ignoring floors or ceilings. These attentional shortcuts dramatically reduce the search space and make human visual search remarkably efficient.


**Reinforcement Learning for Navigation**

Traditional approaches to visual navigation using reinforcement learning typically rely on end-to-end training where agents learn to map raw visual inputs directly to actions. In our implementation, we see this in the baseline train_with_progress_logging.py script, which uses a standard PPO algorithm with CNN-based policies to learn search behaviors. While effective, these approaches often require millions of environment interactions and struggle with the exploration-exploitation dilemma. Agents must spend significant time exploring to discover rewards, leading to inefficient learning, especially in large, complex environments with sparse rewards.

**Gaze as Guidance** 

The theoretical foundation for using human gaze data comes from the insight that gaze patterns contain valuable information about task-relevant features and regions. By incorporating gaze heatmaps as an additional signal, we can provide agents with "attentional priors" that focus learning on important areas of the visual field. Our project implements this through the GazeEnvWrapper and GazePreprocessEnvWrapper classes in env_wrappers.py, which augment observations with gaze prediction data and even modify the reward function to encourage attention to relevant regions.

**Related Work**

This approach builds on a rich body of research using human data to guide AI systems. Imitation learning uses expert demonstrations to bootstrap agent performance, while inverse reinforcement learning attempts to recover reward functions from human behavior. Recent work has also explored using gaze data for various AI tasks, including image classification, video game playing, and autonomous driving. Our project extends these ideas to 3D navigation tasks, implementing three distinct integration methods (channel-based, attention-based, and weighted approaches) to effectively incorporate gaze information into the reinforcement learning pipeline, as seen in our networks.py implementation.

---

#### **Technical Approach**

##### **Overall Architecture:**

The system architecture consists of three integrated components working in harmony:

- **Gaze Prediction Model:** A deep learning model that predicts human attention patterns for visual inputs
- **Environment:** A modified AI2-THOR environment with custom wrappers that integrate gaze information
**RL Agent:** A customized PPO-based agent that learns to navigate using both visual inputs and gaze information

The information flows through this system sequentially - the environment provides RGB observations, which are fed to the gaze model to predict attention heatmaps. These heatmaps are incorporated into the agent's observations, which the agent then processes to make navigation decisions. The reward function uses both search success and gaze-object overlap to guide learning effectively.

##### **Gaze Prediction Model:** 

**Data Collection and Processing**

The gaze prediction model is trained on the SALICON dataset, a large-scale collection of human eye fixation data. The processing pipeline converts raw eye fixation points into smooth attention heatmaps through several steps:

- Creating a zero-initialized heatmap matrix for each image
- Marking human fixation locations with higher values
- Applying Gaussian smoothing to create continuous attention distributions
- Normalizing the results to create proper probability distributions

This preprocessing transforms discrete eye tracking data points into continuous heatmaps that represent where humans typically look when viewing each scene.

**Model Architecture**

The primary gaze prediction architecture uses a ResNet-based model, implementing transfer learning by starting with a pretrained ResNet18 backbone. The model is modified by replacing the final classification layer with a custom regression head that outputs a 224×224 heatmap. A sigmoid activation ensures values are between 0 and 1, representing attention probabilities across the visual field.

**Training Framework**

The training is managed using PyTorch Lightning, which provides a clean, organized structure for model development. The implementation uses Mean Squared Error (MSE) for training and evaluates using both MSE and Structural Similarity Index (SSIM) metrics. All hyperparameters are configured via YAML files, allowing for systematic experimentation with different training configurations.

##### **Environment Setup:**

**AI2-THOR Simulator**

Our implementation wraps the AI2-THOR simulator in a Gymnasium-compatible environment (`AI2ThorEnv` class), providing a standardized interface for reinforcement learning. Key configuration parameters include:

- 224×224 pixel observations
- 0.25m grid size for navigation
- 90° field of view
- Depth and segmentation rendering for richer observations

**Object Search Task Design**

The object search task is implemented in the `reset()` and `step()` methods of our environment class. We create challenging scenarios by:

- Randomly selecting a kitchen scene from a predefined list
- Randomizing the agent's starting position
- Placing target objects (e.g., "Microwave") in their natural locations
- Defining success as finding the object with sufficient visibility (>5%) and proximity (<1.5m)

The episode terminates when either the object is found or the maximum step limit (default: 200 steps) is reached.

**Observation and Action Spaces**

The observation space consists of RGB images (224×224×3) from the agent's perspective. The action space is discrete with 7 possible actions: moving in four directions (forward, backward, left, right), rotating left and right, and looking up. This action space provides sufficient flexibility for the agent to search the environment effectively.

##### **Reinforcement Learning Framework:**

**PPO Algorithm**

The implementation extends Stable-Baselines3's Proximal Policy Optimization (PPO) algorithm with custom feature extractors designed to process gaze information. It supports configurable network architectures for different gaze integration methods and uses hyperparameters specifically tuned for object search tasks.

**Reward Structure**

One of the most innovative aspects of the approach is the reward structure that incorporates gaze information. The reward function includes:

1. A base reward for traditional navigation and search success
2. An additional reward component based on how well the agent's attention (via the gaze model) aligns with relevant objects
3. A mechanism that calculates Intersection over Union (IoU) between predicted gaze and object regions

The base reward structure also includes several components:

- A small penalty for each step to encourage efficiency
- A larger penalty for repeated actions (e.g., bumping into walls)
- Graduated rewards for making the target object visible, with higher rewards for closer and more visible objects
- A large bonus for successfully completing the task

This creates a dense reward signal that guides learning more effectively than sparse rewards would.

**Learning Process**

Our training process is managed by the `train_agent` function in `train_gaze_guided_rl_final.py`, which:

1. Creates a unique experiment directory with timestamp
2. Configures the agent with the specified gaze integration method
3. Trains the agent for the specified number of timesteps
4. Logs comprehensive metrics through our GazeProgressCallback
5. Saves the trained model and performance metrics

The callback tracks:

- Episode rewards and lengths
- Success rates for finding objects
- Overall exploration coverage
- Training speed and resource usage

By combining these components, we create a comprehensive system for gaze-guided reinforcement learning that significantly improves visual search efficiency.

---

#### **Integration Methods (Our Key Innovation)**

Our project explores three distinct methods for integrating gaze information with visual inputs for reinforcement learning. Each method represents a different approach to combining attention data with RGB observations.

##### **Method 1: Channel Integration**

**Technical Approach**

The channel integration method is the most straightforward approach, treating gaze information as a fourth channel alongside the standard RGB channels:

>RGB Image (3 channels) + Gaze Heatmap (1 channel) → 4-channel input

We implemented this in the `ChannelCNN` class by modifying the first convolutional layer of ResNet18:

```python
 if use_gaze:
    original_weights = self.backbone.conv1.weight.data
    self.backbone.conv1 = nn.Conv2d(
        4, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    
    with torch.no_grad():
        self.backbone.conv1.weight.data[:, :3] = original_weights
        self.backbone.conv1.weight.data[:, 3] = original_weights[:, 0]
```

**Implementation Challenges and Solutions**

A major challenge was preserving the pretrained weights while adding the new channel. Our solution was to:

1. Save the original weights for the RGB channels
2. Replace the first convolutional layer with a new 4-channel version
3. Copy the original weights back for the RGB channels
4. Initialize the gaze channel weights using the weights from the red channel

This approach allows us to leverage transfer learning from ImageNet while adapting to our 4-channel input format.

Another challenge was ensuring proper normalization of the gaze channel to match the scale of RGB inputs. We solved this by normalizing gaze heatmaps to the range [0, 255] and then scaling them to [0, 1] during preprocessing, matching the normalization of the RGB channels.

**Theoretical Advantages**

Channel integration offers several advantages:

- **Simplicity:** The approach is conceptually straightforward and easy to implement
- **Early fusion:** Gaze information influences feature extraction from the very beginning
- **Preservation of spatial relationships:** The spatial correlation between RGB and gaze data is maintained
- **Computational efficiency:** No additional network branches or complex fusion mechanisms are required

The main theoretical basis is that early fusion allows the convolutional layers to learn correlations between visual features and attention patterns from the beginning of the processing pipeline.

##### **Method 2: Bottleneck Integration**

**Technical Approach** 

The bottleneck integration method (implemented as GazeAttnCNN) processes RGB and gaze separately before combining them:

>RGB → CNN → RGB Features

>Gaze → Lightweight CNN → Gaze Features

>RGB Features + Gaze Features (via attention) → Fused Features

We implemented a cross-attention mechanism:

```python 
# Process RGB and gaze separately
rgb_feats = self.rgb_backbone(x)
gaze_feats = self.gaze_encoder(gaze_heatmap)

# Use gaze features as query, RGB features as key and value
q = self.query_proj(gaze_feats)
k = self.key_proj(rgb_feats)
v = self.value_proj(rgb_feats)

# Calculate attention and apply to values
attn = torch.matmul(q, k.transpose(-2, -1)) * self.attention_scale
attn = F.softmax(attn, dim=-1)
out = torch.matmul(attn, v)
```

**Implementation Details**

The architecture consists of:

- **RGB Backbone:** ResNet18 up to the final pooling layer
- **Gaze Encoder:** A lightweight CNN with fewer layers
- **Cross-Attention Module:** Projects features to query, key, and value spaces
- **Output Projection:** Global pooling and final feature projection

This architecture allows the network to first extract features from RGB and gaze independently, then use gaze features to guide the attention on RGB features, similar to how human attention works.

**Theoretical Advantages**

Bottleneck integration offers several advantages:

1. **Specialized Processing:** Different network branches can specialize in RGB and gaze feature extraction
2. **Controlled Information Flow:** The attention mechanism explicitly controls how gaze information influences visual features
3. **Interpretable Intermediate Representations:** The attention weights reveal which parts of the RGB features are emphasized
4. **Flexibility:** The architecture can be adapted to different attention mechanisms (multi-head, self-attention, etc.)

The theoretical basis is that attention mechanisms are well-suited for modeling the relationship between gaze and visual features, as they naturally capture the notion of focusing on specific regions.

##### **Method 3: Weighted Integration**

**Technical Approach**

The weighted integration method (WeightedCNN) uses gaze as a spatial modulator for the RGB input:

>Gaze → Gaze Processor → Attention Weights

>RGB * Attention Weights → Modulated RGB → CNN → Features

Our implementation processes the gaze heatmap to generate per-pixel weights:

```python 
# Gaze processor network
self.gaze_processor = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 3, kernel_size=1),  # 3 channels to match RGB
    nn.Sigmoid()  # Outputs weights between 0 and 1
)

# Forward pass
attention_weights = self.gaze_processor(gaze_heatmap)
modulated_input = attention_weights * x
features = self.backbone(modulated_input)
```

**Implementation Details**

The key components are:

1. **Gaze Processor:** A small CNN that converts the gaze heatmap to 3-channel weights
2. **Input Modulation:** Element-wise multiplication of RGB inputs with weights
3. **Backbone CNN:** Standard ResNet18 that processes the modulated input

The sigmoid activation ensures weights are between 0 and 1, effectively acting as attention gates for each pixel and channel.

**Theoretical Advantages**

Weighted integration offers several advantages:

- **Biological Plausibility:** Most closely mimics how human visual attention modulates visual processing
- **Input-Level Integration:** Allows the standard CNN to process "pre-attended" inputs
- **Preservation of Architecture:** Works with any CNN backbone without structural changes
- **Selective Enhancement:** Can enhance or suppress different regions and features based on gaze

The theoretical foundation is that attention acts as a filter or gain control mechanism in human visual processing, enhancing relevant signals and suppressing irrelevant ones before detailed processing.

##### **Comparison and Insights**

Each integration method represents a different hypothesis about how gaze information should influence visual processing:

- **Channel Integration:** Gaze as an additional visual feature
- **Bottleneck Integration:** Gaze as a guide for feature selection
- **Weighted Integration:** Gaze as an input modulator

Our experiments showed that all three methods improved over the baseline, but weighted integration consistently performed best, suggesting that early modulation of visual input is most effective for guiding visual search. 

---

#### **Experimental Setup and Metrics**

**Environments:** Types of rooms and object search tasks

**Training Configuration:** Number of steps, episodes, hyperparameters

**Evaluation Metrics:**

- Success rate
- Efficiency (steps to completion)
- Sample efficiency (learning curve)
- Comparison methodology

---

#### **Results and Analysis**

**Performance Comparison:** How each integration method performed against the baseline

**Sample Efficiency:** Learning curves showing training progress

**Ablation Studies:** Effects of different components on performance

**Qualitative Analysis:** Visual examples of agent behavior with/without gaze guidance

**Insights and Interpretations:** What the results tell us about attention in reinforcement learning

---

#### **Challenges and Solutions**

**Technical Challenges:** Issues you encountered with environment, gaze integration, etc.

**Solutions:** How you overcame these challenges

**Lessons Learned:** What you discovered about the development process

---

#### **Future Directions**

**Potential Improvements:** How your approach could be enhanced

**Applications:** Where this technology could be applied

**Research Possibilities:** New questions opened by your work

**Scaling Considerations:** How this approach might work in more complex environments

---

#### **Conclusion**

**Summary of Contributions:** The key innovations in your work

**Broader Impact:** How this research contributes to the field

**Take-Home Message:** The most important insight from your project

---

#### **Resources**

- **Code Repository:** Link to GitHub
- **Paper/Documentation:** Any publications or documentation
- **Demo:** Any videos or interactive demonstrations
- **References:** Key papers and resources that informed your work

---

#### **Visual Elements to Include**

- System Architecture Diagram: Overall flow from gaze prediction to RL agent
- Integration Method Diagrams: Visual representations of the three integration approaches
- Learning Curves: Comparison of sample efficiency across methods
- Heat Maps: Visualizations of gaze predictions and agent attention
- Code Snippets: Key implementation details
- Agent Navigation Paths: Visual examples of improved paths with gaze guidance