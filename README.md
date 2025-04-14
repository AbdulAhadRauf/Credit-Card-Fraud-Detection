# Credit Card Fraud Detection using Machine Learning

##Overview

This project focuses on building and evaluating machine learning models to detect fraudulent credit card transactions. Credit card fraud is a significant issue for financial institutions and customers alike, leading to substantial financial losses. Detecting fraudulent transactions is challenging due to the highly imbalanced nature of the data – fraudulent transactions are typically very rare compared to legitimate ones.

This repository contains a Jupyter Notebook that walks through the process of exploring the dataset, preprocessing the data, training different classification models (with a focus on K-Nearest Neighbors), and evaluating their performance using appropriate metrics for imbalanced datasets.

## Dataset

*   **Source:** The dataset used is the "Credit Card Fraud Detection" dataset, originally from Kaggle, provided by an anonymous source associated with ULB (Université Libre de Bruxelles).
*   **Content:** It contains anonymized credit card transactions made by European cardholders over two days in September 2013.
*   **Features:**
    *   `Time`: Seconds elapsed between each transaction and the first transaction in the dataset (Note: This feature is dropped during preprocessing in the notebook).
    *   `V1` through `V28`: Anonymized numerical features, which are the result of a Principal Component Analysis (PCA) transformation. Due to confidentiality issues, the original features and more background information about the data are not provided.
    *   `Amount`: The transaction amount.
    *   `Class`: The target variable (response variable), which is 1 in case of fraud and 0 otherwise.
*   **Size:** The dataset contains 284,807 transactions.
*   **Imbalance:** The dataset is highly unbalanced. Only 492 transactions (approximately 0.1727%) are fraudulent, while the remaining 284,315 are legitimate. This imbalance presents a significant challenge for standard classification algorithms and requires careful handling and evaluation.
*   **Missing Values:** The EDA section confirms that there are no missing values in the dataset.

## Installation and Setup

To run the Jupyter Notebook, you need a Python environment with several data science libraries installed.

1.  **Prerequisites:**
    *   Python (Version 3.x recommended)
    *   pip (Python package installer)
    *   Jupyter Notebook, Jupyter Lab, Google Colab, or a similar environment.

2.  **Required Libraries:**
    *   `pandas`: For data manipulation and loading CSV files.
    *   `numpy`: For numerical operations.
    *   `seaborn` & `matplotlib`: For data visualization.
    *   `scikit-learn`: For machine learning tasks (splitting data, scaling, models, metrics).
    *   `imbalanced-learn`: For handling imbalanced datasets (installed in the notebook, though specific resampling techniques like SMOTE or undersampling are not applied to the final evaluated models in this version).

3.  **Installation:** You can install the necessary libraries using pip:
    ```
    pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn jupyterlab
    ```
    *(Alternatively, you can create a `requirements.txt` file and use `pip install -r requirements.txt`)*

## Usage

1.  **Clone or Download:** Get the `main.ipynb` file.
2.  **Dataset Path:** Ensure the path to the `creditcard.csv` dataset file is correctly specified within the notebook. The notebook currently uses:
    ```
    path = "/kaggle/input/creditcardfraud/creditcard.csv"
    ```
    If you are running locally or in a different environment (like Google Colab), you will need to update this `path` variable to point to the location where you have saved the `creditcard.csv` file.
3.  **Run the Notebook:** Open the `.ipynb` file in your chosen Jupyter environment (Jupyter Lab, Google Colab, etc.) and run the cells sequentially from top to bottom.

## Methodology / Workflow

The notebook follows a standard machine learning workflow:

1.  **Data Loading:** The `creditcard.csv` file is loaded into a pandas DataFrame.
2.  **Exploratory Data Analysis (EDA):**
    *   Initial inspection using `head()`, `info()`, `describe()`.
    *   Checked for missing values using `isnull().sum()` (none found).
    *   Analyzed the distribution of the target variable `Class` using `value_counts()` and visualized it with `seaborn.countplot()`. This highlighted the severe class imbalance (0.17% fraud).
3.  **Data Preprocessing:**
    *   **Feature Selection:** The `Time` column was dropped, potentially because it might not be a strong predictor or could introduce temporal dependencies not handled by the models used. `V1`-`V28` and `Amount` were kept as features (X). `Class` was designated as the target (y).
    *   **Train-Test Split:** The data was split into training (70%) and testing (30%) sets using `train_test_split` from scikit-learn. Crucially, `stratify=y` was used to ensure that the proportion of fraudulent and non-fraudulent transactions was maintained in both the training and testing sets. `random_state=42` was used for reproducibility.
    *   **Feature Scaling:** `StandardScaler` from scikit-learn was used *specifically for the K-Nearest Neighbors (KNN) model*. The scaler was `fit` **only** on the training data (`X_train`) to prevent data leakage from the test set, and then used to `transform` both the training (`X_train_scaled`) and testing data (`X_test_scaled`). Decision Trees generally do not require feature scaling.
4.  **Model Training and Evaluation:**
    *   **Model Consideration:** The notebook initially trains and evaluates a `DecisionTreeClassifier` (using `criterion='entropy'`) on the unscaled data.
    *   **KNN Implementation:**
        *   A `KNeighborsClassifier` is trained on the *scaled* training data (`X_train_scaled`).
        *   Initial evaluation is performed with `n_neighbors=1` (K=1).
        *   **Optimal K Selection (Elbow Method):** To find a potentially better value for K, the error rate (1 - accuracy) was calculated for K values ranging from 1 to 39. The error rates were plotted against K values. The "elbow point" in the plot, where the error rate starts to level off, suggests a good balance between bias and variance. The plot generated indicated an optimal K around 3.
5.  **Evaluation Metrics:** Due to the high class imbalance, accuracy alone is a misleading metric. The models were evaluated using:
    *   **Confusion Matrix:** To visualize the counts of true positives, true negatives, false positives, and false negatives.
    *   **Classification Report:** Provides key metrics per class:
        *   **Precision:** (True Positives) / (True Positives + False Positives) - Of all transactions predicted as fraud, how many actually were fraud? (Minimizing false positives is important to avoid inconveniencing legitimate users).
        *   **Recall (Sensitivity):** (True Positives) / (True Positives + False Negatives) - Of all actual fraud transactions, how many were correctly identified? (Maximizing recall is crucial for catching fraud).
        *   **F1-Score:** The harmonic mean of Precision and Recall (2 * Precision * Recall) / (Precision + Recall) - Provides a single score balancing precision and recall.
    *   **Overall Accuracy:** While calculated, it's less informative here due to imbalance.

## Results

*   **Decision Tree:** The Decision Tree classifier achieved high overall accuracy (~99.9%), but this is expected with imbalanced data. For the fraud class (1), it showed:
    *   Precision: ~0.76
    *   Recall: ~0.74
    *   F1-Score: ~0.75
*   **KNN (K=1):** The KNN classifier with K=1, trained on scaled data, showed slightly improved performance for the fraud class compared to the Decision Tree in this run:
    *   Precision: ~0.86
    *   Recall: ~0.76
    *   F1-Score: ~0.81
*   **KNN (Optimal K ≈ 3):** The Elbow Method analysis suggested that K=3 is likely a better choice for the KNN model on this dataset, potentially offering better generalization than K=1 by reducing sensitivity to noise. While the notebook calculates this optimal K, it primarily displays the detailed metrics for K=1. Evaluating the model with K=3 would provide its specific precision/recall/F1 scores. The goal with K=3 is to maintain good Recall (detecting actual fraud) while potentially improving Precision (reducing false alarms on legitimate transactions) compared to some other models or K values.

## Conclusion

This project demonstrated the application of machine learning techniques for credit card fraud detection on a highly imbalanced dataset.

*   EDA revealed the critical challenge of class imbalance.
*   Preprocessing steps like stratified splitting and feature scaling (for distance-based algorithms like KNN) were essential.
*   The Elbow Method provided a systematic way to optimize the hyperparameter K for the KNN model, suggesting K=3 as a potentially optimal value.
*   Evaluation using metrics like Precision, Recall, and F1-Score, along with the Confusion Matrix, is crucial for understanding model performance on imbalanced classification tasks, rather than relying solely on accuracy.

## Future Work

*   Train and explicitly evaluate the KNN model using the optimal K found (K=3) and report its metrics.
*   Implement and compare results using techniques specifically designed for imbalanced data, such as:
    *   Oversampling the minority class (e.g., SMOTE - Synthetic Minority Over-sampling Technique from `imblearn`).
    *   Undersampling the majority class (e.g., `RandomUnderSampler` from `imblearn`).
    *   Using ensemble methods that handle imbalance well (e.g., BalancedRandomForestClassifier, EasyEnsembleClassifier).
*   Experiment with other classification algorithms (e.g., Logistic Regression, Support Vector Machines (SVM), Random Forest, Gradient Boosting) and tune their hyperparameters.
*   Perform more in-depth feature engineering or analysis, although the PCA features limit interpretability. Analyze the 'Amount' feature more closely.
