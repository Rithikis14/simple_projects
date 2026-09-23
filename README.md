# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** Power House
**Team Members:** Rithik V Kumar, Veerabutharan J, vishali Vincent, Suren K
**Submission Date:** 12/10/2025

---

## 1. Executive Summary
*Provide a brief 2-3 sentence overview of your approach and key innovations.*

Our solution employs a systematic, iterative approach, beginning with a simple text-based regression model and progressively enhancing it with targeted, data-driven feature engineering. The key innovation was an error-analysis-driven methodology that identified the baseline model's weaknesses with bulk and specialty items, leading to the creation of custom features that produced our final, best-performing model.

---

## 2. Methodology Overview

### 2.1 Problem Analysis
*Describe how you interpreted the pricing challenge and key insights discovered during EDA.*

We interpreted the challenge as a classic regression problem where the target variable (`price`) was highly non-linear, making direct prediction difficult. The primary challenge was to extract and quantify meaningful price-driving signals from the unstructured `catalog_content` and supplementary `image_link` data.

**Key Observations:**
- **Target Variable Skew:** The `price` distribution was found to be heavily right-skewed. To mitigate this, we applied a logarithmic transformation (`log(1+x)`) to the target variable for training. This resulted in a more normal distribution, which significantly improved model stability and performance.
- **Textual Clues:** The `catalog_content` was identified as a rich source of implicit features, particularly for item quantity (e.g., "pack of," "per case," "lbs") and product type, which were crucial for accurate pricing.
- **Model Performance Puzzle:** Initial experiments revealed that a simple linear model on text features outperformed a more complex multimodal (text + image) model. This indicated that the text contained a very strong signal and that targeted feature engineering would be more impactful than simply adding architectural complexity.

### 2.2 Solution Strategy
*Outline your high-level approach (e.g., multimodal learning, ensemble methods, etc.)*

**Approach Type:** Feature Engineering-Centric Regression
**Core Innovation:** Our core innovation was **Error-Driven Feature Engineering**. Instead of blindly adding complexity, we analyzed the baseline model's largest prediction errors to diagnose its specific weaknesses. This led directly to the creation of high-impact binary features (`is_bulk`, `is_specialty`) that captured concepts the model was missing, resulting in a significant and decisive performance breakthrough.

---

## 3. Model Architecture

### 3.1 Architecture Overview
*Describe your model architecture with a simple diagram or flowchart if possible.*

Our final, champion architecture is a refined `Ridge` regression pipeline. It processes raw `catalog_content` and our custom-engineered numerical features through a parallel preprocessing pipeline (`ColumnTransformer`) before feeding them into the linear model for prediction.

![Architecture Diagram](wiki sumarizer/Input Data (catalog_content, ipq, is_bulk, is_specialty) - visual selection.png)

### 3.2 Model Components

**Text Processing Pipeline:**
- **Preprocessing steps:** Log transformation of the target variable, custom `ipq` extraction from text, and custom binary flag creation for `is_bulk` and `is_specialty` items.
- **Model type:** `TfidfVectorizer` (for text) + `StandardScaler` (for numeric features) feeding into a `Ridge` Regression model.
- **Key parameters:** `TfidfVectorizer(max_features=15000)`, `Ridge(alpha=1.0)`.

**Image Processing Pipeline:**
*Note: This pipeline was part of an exploratory, more complex model that was ultimately outperformed by our refined text-only model.*
- **Preprocessing steps:** Images were resized to 256x256, center-cropped to 224x224, and normalized using standard ImageNet statistics.
- **Model type:** Pre-trained `ResNet-50` used as a feature extractor to generate 2048-dimension embeddings.
- **Key parameters:** The final classification layer of `ResNet-50` was removed to access the feature vectors.

---

## 4. Model Performance

### 4.1 Validation Results
- **SMAPE Score:** **56.3604%** (Achieved by the upgraded Ridge model on the validation set).
- **Other Metrics:** *The following metrics are from a similar baseline run, providing context for the model's performance:*
    - **MAE:** ~$13.79 (The model's predictions are, on average, off by about $14)
    - **RMSE:** ~$28.91 (Indicates the model makes some larger errors on outlier products)
    - **R²:** ~0.17 (The model explains about 17% of the price variance)

## 5. Conclusion
*Summarize your approach, key achievements, and lessons learned in 2-3 sentences.*

Our final solution is a robust and interpretable Ridge regression model, whose success was driven by a meticulous error-analysis-led feature engineering process. The key lesson learned was the immense value of a strong baseline and the principle that targeted feature engineering, guided by a model's specific weaknesses, can often outperform more complex, "black-box" architectures. This proves that a deeper understanding of the data is frequently more valuable than pure model complexity.

---

## Appendix

### A. Code artefacts
*Include drive link for your complete code directory*

*[Insert your Google Drive or GitHub link here]*

### B. Additional Results
*Include any additional charts, graphs, or detailed results*

The full project notebook contains all iterative results, including performance of the multimodal LightGBM model, hyperparameter tuning logs, ensembling experiments, and the detailed error analysis that led to our final model design.

---

