---
layout: page
title: MTA Transit Time Prediction
description: Leveraging real-time data and machine learning to predict bus arrival times in New York City with route-based and grid-based approaches.
img: assets/img/project_1/MTA_1.jpeg
importance: 1
category: work
related_publications: false
---


Ever wondered how buses in New York City could have better estimated arrival times (ETAs)? With the bustling streets, unpredictable traffic, and sheer volume of buses, predicting ETAs is no small feat. Our team embarked on a journey to tackle this challenge by combining live bus location data and real-time traffic information to make transit time predictions smarter and more reliable.

---

#### **The Challenge**

New York City’s transit system is dynamic and complex. Traffic patterns shift with the time of day, weather, and even unexpected events like road closures. Riders often face uncertainty about when their bus will actually arrive.

The goal? To use real-time data from the **MTA BusTime API** and the **TomTom Traffic API** to predict arrival times accurately, even in such a chaotic environment.

---

#### **How We Did It**

We approached the problem using two complementary methods, each tailored to a specific aspect of transit prediction:

##### **1. Route-Based Predictions**
We treated individual bus routes as sequences of stops. Each stop segment—the stretch between two consecutive stops—formed the basis of our analysis. Temporal dependencies (e.g., travel time patterns) were captured using **Long Short-Term Memory (LSTM) networks**, a type of Recurrent Neural Network (RNN) known for handling sequential data effectively.

**Key Details:**
- **Dataset:** Each input sequence had six stops, with 20 features per stop.
- **Architecture:** LSTM units followed by a Dense layer for final ETA prediction.
- **Loss Function:** Mean Squared Error (MSE).
- **Optimizer:** Adam.

Despite LSTMs being state-of-the-art for time-series problems, short sequence lengths (six stops) limited their effectiveness in capturing long-term dependencies.

##### **2. Grid-Based Predictions**
Recognizing that traffic conditions can vary significantly across the city, we divided New York City into grids of varying sizes (e.g., 10×10, 50×50). Each grid cell acted as a mini-region, with its own traffic model trained on localized data.

**Key Details:**
- **Spatial Matching:** We matched bus locations to traffic data using a k-d tree algorithm.
- **Grid Sizes:** Tested grid dimensions from 5×5 (larger cells) to 50×50 (finer cells).
- **Modeling:** XGBoost, a gradient-boosted tree algorithm, performed best for grid-based predictions.
- **Thresholds:** At least 100 data points per grid were required for reliable training.

<div class="row justify-content-sm-center">
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project_1/nest100_5x5.png" title="nest100_5x5" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project_1/nest100_20x20.png" title="nest100_20x20" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project_1/nest100_30x30.png" title="nest100_30x30" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Heatmap showing RMSE for various grid sizes. 1-5x5, 2-20x20, 3-30x30
</div>

---

#### **Building the Dataset**

Creating a unified dataset was critical to our success. Here’s how we merged diverse data sources:

##### **MTA BusTime API**
This API tracked bus locations every minute for 11 days, yielding ~3.5 million raw data points. Each bus’s ETA was calculated by determining when it reached a stop, using the following steps:
1. Identify when the bus was "at the stop" (e.g., distance ≤ 50m or marked as "at stop").
2. Compute the time difference between the recorded time and arrival time.


<div class="row justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project_1/zoomedinroutes.png" title="zoomedinroutes" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption text-center">
    Map of bus routes and stops with data points overlayed
</div>


##### **TomTom Traffic API**
We used real-time traffic incident data to enrich the bus dataset. Each record included:
- Delay magnitude.
- Event descriptions.
- Latitude and longitude of incidents.

To integrate traffic data, we:
1. Aggregated incidents over time to reduce noise.
2. Used spatial matching to associate traffic incidents with nearby bus locations.

##### **Feature Engineering**
Key features were engineered to enhance model performance:
- **Temporal Features:** Hour of the day, day of the week, weekend indicator, rush-hour status.
- **Traffic Features:** Magnitude of delays and proximity to incidents.
- **Categorical Variables:** Encoded attributes like bus ID and stop name.
- **Numerical Scaling:** Standardized all numerical features for consistency.

<div class="row justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project_1/Correlation_Matrix_Processed_Data.png" title="Correlation_Matrix_Processed_Data" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption text-center">
    Correlation matrix of features
</div>

---

#### **Results and Insights**

Our experiments revealed fascinating insights into the strengths and limitations of each approach.

##### **Route-Based Models (LSTMs)**
We trained multiple LSTM architectures to predict travel times. The best-performing model was a simple architecture:
- **Masking → LSTM (32 units) → Dense Layer**
- **Validation RMSE:** 132.23 seconds.

While LSTMs are powerful for sequential data, the short sequence length (six stops) limited their ability to capture long-term dependencies effectively.


##### **Grid-Based Models (XGBoost)**
The grid-based approach performed better overall, leveraging localized traffic patterns. Finer grids (e.g., 50×50) provided the most accurate predictions:
- **RMSE:** 82.02 seconds (50×50 grid).
- **Challenges:** Smaller grids risked insufficient data for training, with ~12% of grids lacking enough points.

---

#### **Comparing Models**

Here’s how our models stacked up:

| **Model**            | **RMSE (s)** | **MAE (s)** | **R² Score** |
|----------------------|--------------|-------------|--------------|
| Linear Regression    | 172.47       | 93.36       | 0.319        |
| Decision Tree        | 142.39       | 61.71       | 0.536        |
| XGBoost (Route-Wise) | 112.04       | 58.67       | 0.713        |
| LSTM                 | 132.23       | —           | —            |
| XGBoost (Grid-Wise)  | 82.02        | 43.73       | —            |

<div class="row justify-content-sm-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project_1/Lasso Regression_plot.png" title="Lasso Regression" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project_1/XGBoost_plot.png" title="XGBoost" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Overlaying the distribution of predicted estimated arrival times (in seconds) on the distribution of actual estimated arrival times shows the Lasso Regression(Left) and XGBoost(Right) generalization and effective learnability of the underlying dataset's distribution.
</div>


<!-- > _[Image Placeholder: Overlayed histograms comparing predicted vs. actual arrival times for LSTM and XGBoost]_ -->

---

#### **What We Learned**

1. **Localized Insights Matter:** The grid-based approach captured nuanced, localized traffic patterns better than route-based models.
2. **Data Structure Drives Model Choice:** While LSTMs excel in sequential tasks, the short sequence lengths limited their effectiveness here.
3. **Feature Engineering Is Key:** Incorporating traffic data and engineered features significantly improved predictions.

---

#### **What’s Next?**

Our work lays the groundwork for smarter transit predictions, but there’s always room to grow:
- **Hybrid Models:** Combine the strengths of route-based and grid-based methods.
- **Incorporate External Data:** Include weather, city events, and real-time crowd density for richer predictions.
- **Expand Beyond NYC:** Apply the model to other cities with varying traffic patterns.

---

#### **Final Thoughts**

Our project underscores the importance of using the right models for the right data. While LSTMs are excellent for sequential tasks, grid-based methods excel in capturing localized patterns in urban settings. By aligning model design with the unique characteristics of urban transit systems, we’re one step closer to making public transportation smarter and more reliable.

**Want to dive deeper? Check out our detailed implementation in the report & code linked below:** 
- <a href="https://drive.google.com/file/d/18ulc0J0qGOSBSYhrQYd6A7KyB27QEwfi/view?usp=sharing">View Full Report (PDF)</a> 
- <a href="https://drive.google.com/file/d/1bXVwFi-551409npz8qCrhw4uiybQTBw5/view?usp=sharing">Code (ZIP)</a>


