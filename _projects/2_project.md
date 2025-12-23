---
layout: page
title: Gaze-Guided Reinforcement Learning for Visual Search
description: Discover how gaze prediction from human eye-tracking enhances AI agents in object search tasks. By integrating visual attention into reinforcement learning through three novel methods, our approach enables faster, more effective navigation in simulated environments.
img: assets/img/proj2-1.jpg
importance: 3
category: Work
related_publications: false
toc:
  sidebar: left
---

<div class="row justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project_2/project-2-intro-pic.png" title="project-2-intro-pic" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

This blog explores how we can make AI agents search for objects more efficiently by mimicking human visual attention patterns. Using a gaze prediction model trained on human eye-tracking data, we've developed three innovative approaches to integrate this visual attention information into a reinforcement learning framework: channel integration (adding gaze as a fourth input channel), bottleneck integration (processing RGB and gaze separately before combining), and weighted integration (using gaze to modulate visual inputs).

Our experiments in simulated indoor environments demonstrate that these gaze-guided agents learn more efficiently and navigate more effectively than traditional approaches. By bridging cognitive science and machine learning, this project offers insights into how biologically-inspired attention mechanisms can enhance AI systems for robotics, assistive technologies, and autonomous navigation.

This project was developed as part of the **Deep Decision Making and Reinforcement Learning** course during the spring 2025 semester in the **MSCS program at NYU Courant**. Our team — **Lokesh Gautham Boominathan Muthukumar**, **Yu Minematsu**, **Muhammad Mukthar**, and myself.

---

#### **Introduction and Motivation**

Visual search is a fundamental task that humans perform effortlessly every day. Whether looking for your keys, finding a product on a shelf, or locating a friend in a crowd, our visual system efficiently guides our attention to relevant objects. However, autonomous agents struggle with these same tasks, often employing inefficient search strategies that waste time and computational resources.

This project explores a novel approach: leveraging human gaze patterns to enhance reinforcement learning for visual search tasks. By integrating human attention data, we aim to teach AI agents to "look where humans look," dramatically improving search efficiency in environments like AI2-THOR, a realistic 3D household simulator.

The motivation behind this research stems from a simple observation: humans use sophisticated attention mechanisms developed through evolution to efficiently prioritize visual information. When we search for objects, our eyes don't scan uniformly across a scene; instead, we rapidly focus on relevant locations based on context, object relationships, and prior knowledge. For instance, when looking for a microwave in a kitchen, humans instinctively focus on countertops and walls rather than floors or ceilings.

While gaze prediction has been explored in computer vision research, and various attention mechanisms have been applied to object detection, our work makes a novel contribution by specifically integrating human gaze patterns into reinforcement learning for embodied visual search in 3D environments. Previous studies have primarily focused on using gaze for image classification, intention prediction, or as auxiliary signals, but our approach represents one of the first comprehensive attempts to integrate gaze information at both perceptual and motivational levels for embodied agents performing active search tasks.

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


<div class="row justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project_2/proj2-pic2.png" title="proj2-pic2" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

**Reinforcement Learning for Navigation**

Traditional approaches to visual navigation using reinforcement learning typically rely on end-to-end training where agents learn to map raw visual inputs directly to actions. In our implementation, we see this in the baseline `train_with_progress_logging.py` script, which uses a standard PPO algorithm with CNN-based policies to learn search behaviors. While effective, these approaches often require millions of environment interactions and struggle with the exploration-exploitation dilemma. Agents must spend significant time exploring to discover rewards, leading to inefficient learning, especially in large, complex environments with sparse rewards.

**Gaze as Guidance** 

The theoretical foundation for using human gaze data comes from the insight that gaze patterns contain valuable information about task-relevant features and regions. By incorporating gaze heatmaps as an additional signal, we can provide agents with "attentional priors" that focus learning on important areas of the visual field. Our project implements this through the GazeEnvWrapper and GazePreprocessEnvWrapper classes in env_wrappers.py, which augment observations with gaze prediction data and even modify the reward function to encourage attention to relevant regions.

**Related Work**

This approach builds on a rich body of research using human data to guide AI systems. Imitation learning uses expert demonstrations to bootstrap agent performance, while inverse reinforcement learning attempts to recover reward functions from human behavior. Recent work has also explored using gaze data for various AI tasks, including image classification, video game playing, and autonomous driving. Our project extends these ideas to 3D navigation tasks, implementing three distinct integration methods (channel-based, attention-based, and weighted approaches) to effectively incorporate gaze information into the reinforcement learning pipeline, as seen in our networks.py implementation.

---

#### **Technical Approach**

The system architecture consists of three integrated components working in harmony:

- **Gaze Prediction Model:** A deep learning model that predicts human attention patterns for visual inputs
- **Environment:** A modified AI2-THOR environment with custom wrappers that integrate gaze information
- **RL Agent:** A customized PPO-based agent that learns to navigate using both visual inputs and gaze information

The information flows through this system sequentially - the environment provides RGB observations, which are fed to the gaze model to predict attention heatmaps. These heatmaps are incorporated into the agent's observations, which the agent then processes to make navigation decisions. The reward function uses both search success and gaze-object overlap to guide learning effectively.

<div class="row justify-content-center">
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project_2/sys-arch.jpeg" title="sys-arch" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

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

<div class="row justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project_2/env-1.png" title="env-1" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The episode terminates when either the object is found or the maximum step limit (default: 200 steps) is reached.

**Observation and Action Spaces**

The observation space consists of RGB images (224×224×3) from the agent's perspective. The action space is discrete with 7 possible actions: moving in four directions (forward, backward, left, right), rotating left and right, and looking up. This action space provides sufficient flexibility for the agent to search the environment effectively.

<div class="row justify-content-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project_2/env-2.png" title="env-2" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

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

##### **Comparison and Insights:**

Each integration method represents a different hypothesis about how gaze information should influence visual processing:

- **Channel Integration:** Gaze as an additional visual feature
- **Bottleneck Integration:** Gaze as a guide for feature selection
- **Weighted Integration:** Gaze as an input modulator

Our experiments showed that all three methods improved over the baseline, but weighted integration consistently performed best, suggesting that early modulation of visual input is most effective for guiding visual search. 

---

#### **Experimental Setup and Metrics**

Our experiments were conducted in the AI2-THOR simulator, which provides photorealistic 3D household environments. Specifically, we focused on kitchen scenes for our object search tasks:

```python
kitchen_scenes = [f"FloorPlan{i}" for i in range(1, 31) if i <= 5 or 25 <= i <= 30]
```

This selection provided us with 11 distinct kitchen layouts with varying complexity and furniture arrangements. Each kitchen contains multiple objects in their natural locations, creating a challenging but realistic search environment.

For our primary experiments, we selected the "Microwave" as the target object, as it:

- Is present in all kitchen scenes
- Can be located in different positions (counter, island, above stove)
- Has distinctive visual features
- Represents a realistic household search target

To ensure robust evaluation, we randomized:

- The starting position of the agent in each episode
- The orientation of the agent (random rotation)
- The specific kitchen scene for each episode

This randomization prevented the agent from memorizing specific paths and forced it to develop generalizable search strategies.

##### **Training Configuration:**

We used the following training configuration as specified in our default.yaml file:

```yaml
# Environment settings
environment:
  scene_type: "FloorPlan"
  target_object: "Microwave"  # Default target
  grid_size: 0.25             # Navigation grid resolution
  rotation_step: 45           # Agent rotation angle per step
  field_of_view: 90           # Agent's camera FOV
  max_steps: 500              # Maximum episode length
  visibility_distance: 1.5    # Object detection threshold
  done_on_target_found: True  # Episode terminates on success
```

For our PPO algorithm, we used these hyperparameters:

```yaml
# Model settings
model:
  # PPO hyperparameters
  lr: 3.0e-4                        # Learning rate
  n_steps: 128                      # Steps per update
  batch_size: 64                    # Mini-batch size
  n_epochs: 4                       # Epochs per update
  gamma: 0.99                       # Discount factor
  gae_lambda: 0.95                  # GAE parameter
  clip_range: 0.2                   # PPO clip range
  ent_coef: 0.01                    # Entropy coefficient
  vf_coef: 0.5                      # Value function coefficient
  max_grad_norm: 0.5                # Gradient clipping
  features_extractor: "WeightedCNN" # {"ChannelCNN", BottleneckCNN", "WeightedCNN"}
  use_gaze: False                   # Toggled for gaze experiments
```

Each training run:

- Used a single environment wrapped with DummyVecEnv for Stable Baselines compatibility
- Trained for 100,000 timesteps per experiment (approximately 200 episodes)
- Used a maximum episode length of 500 steps
- Logged detailed metrics including episode rewards, success rates, and navigation efficiency
- Used TensorBoard for visualizing learning progress
- Created unique experiment directories for each run with timestamped IDs

For our gaze integration research, we conducted experiments with:

1. Baseline agent (standard PPO with no gaze information)
2. Channel integration (gaze as 4th input channel)
3. Bottleneck integration (separate RGB and gaze processing paths)
4. Weighted integration (gaze-modulated RGB input)

All configurations were evaluated across three environment types with increasing complexity to test generalization capabilities and the impact of visual distractions on search efficiency.

##### **Evaluation Metrics:**

We used several metrics to evaluate the performance of our agents:

**Success Rate**

The primary metric was success rate, defined as the percentage of episodes where the agent successfully found the target object. Success was determined by:

- The target object being visible in the agent's field of view
- The agent being within 1.5 meters of the object
- The object occupying at least 5% of the agent's visual field

This success definition ensured that the agent not only found the object but was also in a position to potentially interact with it.

**Efficiency (Steps to Completion)**

We measured the average number of steps taken to find the target object in successful episodes. This metric evaluates the efficiency of the search strategy, with fewer steps indicating a more direct path to the target.

**Exploration Coverage**

We tracked the number of unique positions visited by the agent during an episode, divided by the total number of navigable positions in the scene. This metric evaluates how thoroughly the agent explores the environment.

```python
"exploration_coverage": len(self.visited_positions) / 100.0  # Normalize by approximate reachable positions
```

**Gaze-Object Alignment**

For gaze-guided methods, we measured the average IoU (Intersection over Union) between the predicted gaze heatmap and object regions. This metric evaluates how well the gaze prediction aligns with actual object locations.

**Comparison Methodology**

To ensure fair comparison between methods, we:

1. Used identical environment configurations for all agents
2. Initialized each agent with the same network architecture (except for gaze integration)
3. Used the same hyperparameters across all methods when possible

For our final evaluation, we tested each trained agent on 50 new episodes with:

- Random starting positions
- Previously unseen kitchen configurations
- Random target object placements

This thorough evaluation methodology allowed us to rigorously compare the different integration methods and determine which approach most effectively leveraged gaze information for visual search.

---

#### **Results and Analysis**

Our experiments across different environments and integration methods yielded several key findings:

**General Performance Trends**

The integration of gaze information significantly improved agent performance across all tested environments. Gaze integration consistently provided a substantial baseline reward increase compared to the standard PPO agent without gaze guidance. This performance gain was evident early in training, suggesting that gaze information provides useful priors for exploration.

<div class="row justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project_2/comparison_plots_train_general_new.png" title="comparison_plots_train_general_new" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
comparison_plots_train_general_new
</div>

Environment complexity played a crucial role in determining the relative advantages of gaze integration:

<div class="row justify-content-center">
    <div class="col-sm-6 mt-3">
        {% include figure.liquid path="assets/img/project_2/comparison_plots_train_floorplan1_all.png" title="comparison_plots_train_floorplan1_all" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-6 mt-3">
        {% include figure.liquid path="assets/img/project_2/comparison_plots_train_floorplan30_all.png" title="comparison_plots_train_floorplan30_all" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
comparison_plots_train_floorplan1_all & comparison_plots_train_floorplan30_all
</div>

**Simple Environments (FloorPlan1):** In simpler floor plans with minimal distractions, the baseline agent learned effective policies relatively quickly. The gaze-integrated models, being more complex, required additional training time to fully leverage their capabilities. However, once trained sufficiently, they still outperformed the baseline.

**Complex Environments (FloorPlan30):** In more complicated environments with numerous visual distractions, the advantage of gaze integration became more pronounced. These models helped agents start with a reasonable prior rather than navigating randomly, significantly reducing the exploration space. The baseline agent often wasted time exploring irrelevant areas of the environment.

##### **Ablation Studies:**

We conducted several ablation studies to understand the contribution of different gaze integration components:

**Gaze Features vs. Gaze Rewards**

<div class="row justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project_2/comparison_plots_train_floorplan30_all_table_case.png" title="comparison_plots_train_floorplan30_all_table_case" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
comparison_plots_train_floorplan30_all_table_case
</div>

Our experiments revealed that gaze features alone did not significantly improve performance. However, when combined with gaze-based reward incentives, performance improved substantially. This suggests a synergistic effect between perception (gaze features) and motivation (reward structure).

**Integration Methods Comparison**

Each integration method showed distinct advantages:

1. **Channel Integration:** The simplest approach, providing solid performance improvements with minimal architectural changes.
2. **Bottleneck Integration:** Offered better feature representation but required more training time to converge.
3. **Weighted Integration:** Showed the most promising results in complex environments by effectively focusing attention on relevant regions.

**Training Duration Effects**

All gaze integration methods required more training timesteps to reach optimal performance compared to the baseline. This reflects the increased model complexity and the need to learn effective representations of the gaze information. However, this investment in training time translated to superior performance, especially in challenging environments.

##### **Qualitative Analysis:**

Visual examination of agent behavior revealed striking differences between baseline and gaze-guided agents:

<div class="container">
  <div class="row justify-content-center">
    <div class="col-12 d-flex justify-content-center">
      <div style="max-width: 960px; width: 50%;">
        {% include video.liquid 
          path="assets/img/project_2/episode-success.mp4" 
          class="img-fluid rounded z-depth-1 w-100" 
          controls=true 
          autoplay=true 
        %}
      </div>
    </div>
  </div>
  <div class="caption text-center mt-2">
    A successful episode of locating the target microwave using gaze guidance.
  </div>
</div>

**Navigation Patterns**

- **Baseline Agents:** Often exhibited wandering behavior, moving in inefficient patterns and frequently revisiting already explored areas.
- **Gaze-Guided Agents:** Demonstrated more directed navigation, with clear purpose in movement and fewer redundant actions.

**Policy Stability**

An unexpected benefit of gaze integration was increased policy stability. The variance in performance across episodes was notably reduced when using gaze guidance, suggesting that human attention patterns provide a stabilizing influence on learning.

<div class="row justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project_2/comparison_plots_eval_floorplan30_all.png" title="comparison_plots_eval_floorplan30_all" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
comparison_plots_eval_floorplan30_all
</div>


##### **Insights and Interpretations:** 

Our results provide several important insights about attention in reinforcement learning:

**The Synergy Effect**

The most significant finding is that gaze features must be paired with appropriate reward incentives to be effective. This reveals a fundamental principle: agents need both a map (where to look) and a reason (why to look there). This parallels human cognition, where attention and motivation are deeply interconnected.

**Complexity-Performance Relationship**

The advantage of gaze guidance scales with environment complexity. In simple environments, standard RL approaches can efficiently discover solutions, but as complexity increases, human-like attention becomes increasingly valuable as a focusing mechanism.

The performance advantages demonstrated by gaze-guided agents, particularly in complex environments, strongly support the hypothesis that human attention patterns can significantly improve reinforcement learning for visual search tasks.

---

#### **Challenges and Solutions**

Implementing gaze-guided reinforcement learning presented several significant technical challenges that required innovative solutions:

**Environment Integration Challenges**

One of the primary challenges was integrating the gaze prediction model with the AI2Thor environment while maintaining compatibility with Stable Baselines 3's PPO implementation. 

AI2Thor's observation space and Stable Baselines' expectations didn't align directly, creating several integration issues:

1. **Observation Space Incompatibility:** AI2Thor provides RGB observations, but we needed to add a gaze channel while keeping everything compatible with Stable Baselines' expected formats.
2. **Reward Signal Modification:** Incorporating gaze-based rewards required intercepting and modifying the environment's reward calculation without disrupting the training loop.
3. **Feature Extractor Complexity:** Each integration method required a custom feature extractor that could process both RGB and gaze information correctly.
4. **Memory Management:** AI2Thor instances consumed substantial memory, causing occasional crashes during extended training runs.

**Gaze Integration Challenges**

The integration of gaze information into the reinforcement learning pipeline presented several architectural challenges:

1. **Neural Network Architecture:** Designing effective architectures for each integration method required balancing complexity with performance.
2. **Training Stability:** The more complex models showed higher variance during early training phases.
3. **Hyperparameter Tuning:** Finding optimal hyperparameters across different integration methods proved challenging.
4. **Feature Representation:** Ensuring that gaze features were properly normalized and represented within the model.

##### **Solutions:** 

We implemented several solutions to address these challenges:

**Environment Wrapper Architecture**

We developed a multi-layered wrapper architecture to seamlessly integrate gaze prediction with the AI2Thor environment:

```python
# Create environment with gaze
env_fn = create_env(config, args.target, gaze_model=gaze_model)
env = DummyVecEnv([env_fn])
```

The `create_env` function implemented a series of wrappers:

- **GazeEnvWrapper:** This wrapper intercepted observations from AI2Thor, ran them through the gaze prediction model, and augmented the observation with a gaze heatmap.
- **GazePreprocessEnvWrapper:** Ensured that observations were properly formatted with the correct shape and channel ordering.
- **FlattenObservationWrapper:** Made the observations compatible with Stable Baselines 3's expectations.

This layered approach allowed us to maintain clean separation of concerns while ensuring compatibility across all components.

**Custom Feature Extractors**

For each integration method, we implemented specialized feature extractors that properly handled the combined RGB and gaze information:

```python
class ChannelGazeExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        self.cnn = CNN(use_gaze=True)
    
    def forward(self, observations):
        # Process observations
        if isinstance(observations, np.ndarray):
            observations = torch.FloatTensor(observations)
        
        # Normalize RGB and extract gaze
        rgb = observations[:, :3].float() / 255.0
        gaze = observations[:, 3:4].float() / 255.0 if observations.shape[1] > 3 else torch.zeros(...)
        
        # Forward pass
        return self.cnn(rgb, gaze)
```

Similar extractors were implemented for bottleneck and weighted integration methods, each handling the observation processing appropriately.

**Reward Augmentation**

To incorporate gaze guidance into the reward structure, we implemented a reward augmentation mechanism:

```python
def _augment_reward(self, reward, observation, success):
    """Augment reward based on gaze alignment with important regions"""
    if success:
        return reward  # Don't modify success reward
    
    # Generate gaze heatmap
    gaze_heatmap = self._predict_gaze(observation)
    
    # Calculate reward based on alignment of agent's view with gaze predictions
    alignment_score = self._compute_alignment(observation, gaze_heatmap)
    
    # Scale and add to base reward
    gaze_reward = self.gaze_reward_scale * alignment_score
    return reward + gaze_reward
```

This allowed us to guide the agent's exploration without interfering with the base environment reward structure.

**Error Handling and Training Management**

To address the stability issues with AI2Thor and extended training runs, we implemented several robustness improvements:

1. **Checkpoint Saving:** Regular checkpoints to resume training after crashes
2. **Error Recovery:** Try-except blocks with clean environment shutdown
3. **Resource Monitoring:** Memory usage tracking and graceful degradation
4. **Training Resumption:** Ability to continue training from saved checkpoints

```python
try:
    model.learn(
        total_timesteps=total_timesteps,
        callback=progress_callback
    )
except Exception as e:
    logger.error(f"\nERROR during training: {e}")
    import traceback
    logger.error(traceback.format_exc())
finally:
    # Clean up
    try:
        if 'env' in locals():
            env.close()
            logger.info("Environment closed")
    except Exception as cleanup_error:
        logger.error(f"Error during cleanup: {cleanup_error}")
```


##### **Lessons Learned:**

Through the development and experimentation process, we discovered several important insights:

- **Training Duration Requirements:** The more complex gaze-integrated models required significantly more training time than our initial estimates. For these complex environments and neural architectures, at least a million timesteps would be necessary to see optimal performance. Our experiments of 100,000 timesteps provided promising initial results but likely undersell the full potential of gaze integration.
- **Simulator Limitations:** AI2Thor, while powerful for realistic indoor navigation, had stability issues during extended runs. These crashes limited our ability to conduct the longer training sessions that would likely further demonstrate the advantages of gaze integration.
- **Observation Preprocessing Importance:** Proper normalization and preprocessing of observations proved critical for stable training, especially with the combined RGB and gaze inputs.
- **Feature Extraction Architecture Impact:** The choice of feature extraction architecture had a significant impact on both training stability and ultimate performance. Simpler architectures trained more quickly but had lower performance ceilings.

Despite the limited training duration, our results showed clear signals that gaze integration provides substantial benefits, particularly in complex environments. With more training time, the performance gap would likely widen further in favor of gaze-guided approaches.


---

#### **Future Directions**

Our work points to several promising avenues for improvement:

**Extended Training Duration**

The most immediate improvement would be to extend training duration to at least 1 million timesteps. Our preliminary results with 50,000 timesteps already showed significant promise, and the learning curves suggested that performance would continue to improve with additional training. More extensive training would likely reveal the full potential of gaze integration, particularly for the more complex integration methods.

**Architectural Refinements**

The neural network architectures could be further refined to better leverage gaze information. More sophisticated attention mechanisms could better integrate gaze with visual features. Transformer-based architectures could be particularly effective for this purpose.

**Improved Gaze Prediction Models**

Our current gaze prediction model could be enhanced in several ways:
- Training gaze models specifically for object search tasks could provide more relevant attention patterns.
-  Incorporating additional modalities like depth information could improve gaze prediction accuracy.


##### **Scaling Considerations:** 

Scaling this approach to more complex environments presents both challenges and opportunities:

- More complex environments and extended training would require significant computational resources. Techniques like distributed training and more efficient implementations would be necessary for large-scale deployment.
- As environments become more complex (e.g., multi-room buildings, outdoor settings), the value of gaze guidance likely increases, but so does the challenge of providing accurate gaze predictions. Research into more sophisticated gaze models would be needed.
- Extending beyond object search to more complex tasks like manipulation or sequential decision-making would require more sophisticated integration of gaze information throughout the task structure.

---

#### **Conclusion**

Our project demonstrates that integrating human visual attention patterns into reinforcement learning creates agents that navigate and search more efficiently. By developing three different integration methods—channel, bottleneck, and weighted, we've shown how gaze information can guide visual search in various environmental contexts.

##### **Key Message:**

>Human attention is a powerful guide for artificial intelligence, but unlocking its full potential requires more than simply showing an agent where to look—it requires teaching the agent why looking there matters. By aligning what an agent sees with what it values through a combination of perceptual guidance and reward shaping, we can create systems that navigate complex visual environments with greater purpose and efficiency. This approach represents a step toward more human-like artificial intelligence that doesn't just mimic human behavior but inherits the underlying cognitive principles that make human visual search so remarkably efficient.

---

#### **Resources**

- **Code Repository:** [Link to GitHub Repository](https://github.com/Monishver11/Gaze-RL)
- **Data/Checkpoints:** [Link to Drive](https://drive.google.com/drive/folders/1zZxV_rkoEhV5sZV9cidnpqYMvq0FwDUe?usp=sharing)
- **Paper/Documentation:** Any publications or documentation
- **References:** Key papers and resources

<!-- #### **Visual Elements to Include**

- System Architecture Diagram: Overall flow from gaze prediction to RL agent
- Integration Method Diagrams: Visual representations of the three integration approaches
- Learning Curves: Comparison of sample efficiency across methods
- Heat Maps: Visualizations of gaze predictions and agent attention
- Code Snippets: Key implementation details
- Agent Navigation Paths: Visual examples of improved paths with gaze guidance -->