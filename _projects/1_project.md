---
layout: page
title: MTA Transit Time Prediction
description: Leveraging real-time data and machine learning to predict bus arrival times in New York City with route-based and grid-based approaches.
img: assets/img/12.jpg
importance: 1
category: work
related_publications: false
---


Ever wondered how buses in New York City could have better estimated arrival times (ETAs)? With the bustling streets, unpredictable traffic, and sheer volume of buses, predicting ETAs is no small feat. Our team embarked on a journey to tackle this challenge by combining live bus location data and real-time traffic information to make transit time predictions smarter and more reliable.

---

#### The Challenge

New York City’s transit system is dynamic and complex. Traffic patterns shift with the time of day, weather, and even unexpected events like road closures. Riders often face uncertainty about when their bus will actually arrive.

The goal? To use real-time data from the **MTA BusTime API** and the **TomTom Traffic API** to predict arrival times accurately, even in such a chaotic environment.

---

#### How We Did It

We approached the problem using two complementary methods, each tailored to a specific aspect of transit prediction:

##### 1. Route-Based Predictions
We treated individual bus routes as sequences of stops. Each stop segment—the stretch between two consecutive stops—formed the basis of our analysis. Temporal dependencies (e.g., travel time patterns) were captured using **Long Short-Term Memory (LSTM) networks**, a type of Recurrent Neural Network (RNN) known for handling sequential data effectively.

**Key Details:**
- **Dataset:** Each input sequence had six stops, with 20 features per stop.
- **Architecture:** LSTM units followed by a Dense layer for final ETA prediction.
- **Loss Function:** Mean Squared Error (MSE).
- **Optimizer:** Adam.

Despite LSTMs being state-of-the-art for time-series problems, short sequence lengths (six stops) limited their effectiveness in capturing long-term dependencies.

##### 2. Grid-Based Predictions
Recognizing that traffic conditions can vary significantly across the city, we divided New York City into grids of varying sizes (e.g., 10×10, 50×50). Each grid cell acted as a mini-region, with its own traffic model trained on localized data.

**Key Details:**
- **Spatial Matching:** We matched bus locations to traffic data using a k-d tree algorithm.
- **Grid Sizes:** Tested grid dimensions from 5×5 (larger cells) to 50×50 (finer cells).
- **Modeling:** XGBoost, a gradient-boosted tree algorithm, performed best for grid-based predictions.
- **Thresholds:** At least 100 data points per grid were required for reliable training.

> _[Image Placeholder: Heatmap showing RMSE for various grid sizes]_

---

#### Building the Dataset

Creating a unified dataset was critical to our success. Here’s how we merged diverse data sources:

##### MTA BusTime API
This API tracked bus locations every minute for 11 days, yielding ~3.5 million raw data points. Each bus’s ETA was calculated by determining when it reached a stop, using the following steps:
1. Identify when the bus was "at the stop" (e.g., distance ≤ 50m or marked as "at stop").
2. Compute the time difference between the recorded time and arrival time.

<div align="center">
    <img src="assets/img/project_1/zoomedinroutes.png" alt="Gradient Descent GIF" width="500">
    <p>Map of bus routes and stops with data points overlayed</p>
</div>

<!-- > _[Image Placeholder: Map of bus routes and stops with data points overlayed]_ -->

##### TomTom Traffic API
We used real-time traffic incident data to enrich the bus dataset. Each record included:
- Delay magnitude.
- Event descriptions.
- Latitude and longitude of incidents.

To integrate traffic data, we:
1. Aggregated incidents over time to reduce noise.
2. Used spatial matching to associate traffic incidents with nearby bus locations.

##### Feature Engineering
Key features were engineered to enhance model performance:
- **Temporal Features:** Hour of the day, day of the week, weekend indicator, rush-hour status.
- **Traffic Features:** Magnitude of delays and proximity to incidents.
- **Categorical Variables:** Encoded attributes like bus ID and stop name.
- **Numerical Scaling:** Standardized all numerical features for consistency.

<div align="center">
    <img src="assets/img/project_1/Correlation_Matrix_Processed_Data.png" alt="Gradient Descent GIF" width="500">
    <p>Correlation matrix of features</p>
</div>

<!-- > _[Image Placeholder: Correlation matrix of features]_ -->

---

#### Results and Insights

Our experiments revealed fascinating insights into the strengths and limitations of each approach.

##### Route-Based Models (LSTMs)
We trained multiple LSTM architectures to predict travel times. The best-performing model was a simple architecture:
- **Masking → LSTM (32 units) → Dense Layer**
- **Validation RMSE:** 132.23 seconds.

While LSTMs are powerful for sequential data, the short sequence length (six stops) limited their ability to capture long-term dependencies effectively.

<!-- > _[Image Placeholder: Diagram of LSTM architecture]_ -->

##### Grid-Based Models (XGBoost)
The grid-based approach performed better overall, leveraging localized traffic patterns. Finer grids (e.g., 50×50) provided the most accurate predictions:
- **RMSE:** 82.02 seconds (50×50 grid).
- **Challenges:** Smaller grids risked insufficient data for training, with ~12% of grids lacking enough points.

---

#### Comparing Models

Here’s how our models stacked up:

| **Model**            | **RMSE (s)** | **MAE (s)** | **R² Score** |
|----------------------|--------------|-------------|--------------|
| Linear Regression    | 172.47       | 93.36       | 0.319        |
| Decision Tree        | 142.39       | 61.71       | 0.536        |
| XGBoost (Route-Wise) | 112.04       | 58.67       | 0.713        |
| LSTM                 | 132.23       | —           | —            |
| XGBoost (Grid-Wise)  | 82.02        | 43.73       | —            |

<!-- <div align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/a/a3/Gradient_descent.gif" alt="Gradient Descent GIF" width="500">
    <p>Overlaying the distribution of predicted estimated arrival times (in seconds) on the distribution of actual estimated arrival times shows that Lasso regression(Left) and XGBoost(Right) generalizes effectively and learns the underlying dataset's distribution.</p>
</div> -->

<div class="row justify-content-sm-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project_1/Lasso Regression_plot.png" title="Lasso Regression" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project_1/XGBoost_plot.png" title="XGBoost" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Overlaying the distribution of predicted estimated arrival times (in seconds) on the distribution of actual estimated arrival times shows that Lasso Regression(Left) and XGBoost(Right) generalizes effectively and learns the underlying dataset's distribution.
</div>


> _[Image Placeholder: Overlayed histograms comparing predicted vs. actual arrival times for LSTM and XGBoost]_

---

#### What We Learned

1. **Localized Insights Matter:** The grid-based approach captured nuanced, localized traffic patterns better than route-based models.
2. **Data Structure Drives Model Choice:** While LSTMs excel in sequential tasks, the short sequence lengths limited their effectiveness here.
3. **Feature Engineering Is Key:** Incorporating traffic data and engineered features significantly improved predictions.

---

#### What’s Next?

Our work lays the groundwork for smarter transit predictions, but there’s always room to grow:
- **Hybrid Models:** Combine the strengths of route-based and grid-based methods.
- **Incorporate External Data:** Include weather, city events, and real-time crowd density for richer predictions.
- **Expand Beyond NYC:** Apply the model to other cities with varying traffic patterns.

---

#### Final Thoughts

Our project underscores the importance of using the right models for the right data. While LSTMs are excellent for sequential tasks, grid-based methods excel in capturing localized patterns in urban settings. By aligning model design with the unique characteristics of urban transit systems, we’re one step closer to making public transportation smarter and more reliable.

**Want to dive deeper? Check out our detailed implementation in the report linked below:** [View Full Report (PDF)](your-pdf-link-here)





<!-- Every project has a beautiful feature showcase page.
It's easy to include images in a flexible 3-column grid format.
Make your photos 1/3, 2/3, or full width.

To give your project a background in the portfolio page, just add the img tag to the front matter like so:

    ---
    layout: page
    title: project
    description: a project with a background image
    img: /assets/img/12.jpg
    ---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/1.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Caption photos easily. On the left, a road goes through a tunnel. Middle, leaves artistically fall in a hipster photoshoot. Right, in another hipster photoshoot, a lumberjack grasps a handful of pine needles.
</div>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    This image can also have a caption. It's like magic.
</div>

You can also put regular text between your rows of images, even citations {% cite einstein1950meaning %}.
Say you wanted to write a bit about your project before you posted the rest of the images.
You describe how you toiled, sweated, _bled_ for your project, and then... you reveal its glory in the next row of images.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    You can also have artistically styled 2/3 + 1/3 images, like these.
</div>

The code is simple.
Just wrap your images with `<div class="col-sm">` and place them inside `<div class="row">` (read more about the <a href="https://getbootstrap.com/docs/4.4/layout/grid/">Bootstrap Grid</a> system).
To make images responsive, add `img-fluid` class to each; for rounded corners and shadows use `rounded` and `z-depth-1` classes.
Here's the code for the last row of images above:

{% raw %}

```html
<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
```

{% endraw %} -->
