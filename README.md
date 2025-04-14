# Credit Card Fraud Detection using Machine Learning


This project dives into the important world of detecting fraudulent credit card transactions using machine learning. As you know, catching fraud is a big deal for both banks and customers to prevent losses. The main challenge? Fraudulent transactions are super rare compared to normal ones, making this a classic case of dealing with **highly imbalanced data**.

This notebook walks you through the whole process: checking out the data, cleaning it up, trying out a couple of classification models (especially K-Nearest Neighbors), and seeing how well they perform using the *right* metrics for this kind of skewed data.

## The Data We're Using

*   **Source:** We're using the popular "Credit Card Fraud Detection" dataset from Kaggle (thanks to the folks at ULB!).
*   **What's Inside:** It's a collection of anonymized transactions from European cardholders over two days back in September 2013.
*   **The Features:**
    *   `Time`: Seconds between transactions (we actually drop this later).
    *   `V1` to `V28`: These are anonymized features, likely the result of PCA (Principal Component Analysis). We don't know the original features due to privacy.
    *   `Amount`: The transaction amount.
    *   `Class`: This is our target! 1 if it's fraud, 0 if it's normal.
*   **Size & Imbalance:** There are 284,807 transactions in total, but only 492 are fraudulent (a tiny **0.17%**!). This imbalance is the key thing we need to keep in mind.
*   **Missing Stuff?:** Nope! A quick check confirms there are no missing values to worry about.

## Getting Started

Ready to run the code? Here's what you'll need:

1.  **Your Environment:**
    *   Python 3.x
    *   Pip (for installing packages)
    *   A way to run Jupyter notebooks (like Jupyter Lab, Jupyter Notebook, Google Colab, VS Code).

2.  **Key Libraries:**
    *   `pandas` (for handling the data)
    *   `numpy` (for number crunching)
    *   `seaborn` & `matplotlib` (for making pretty plots)
    *   `scikit-learn` (the powerhouse for ML tasks - splitting data, scaling, models, metrics)
    *   `imbalanced-learn` (useful tools for imbalanced data, though we don't use its advanced sampling in this *specific* notebook's final models)

3.  **Installation:** Get the libraries using pip:
    ```
    pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn jupyterlab
    ```
    *(run `pip install -r requirements.txt`)*

## How to Run It

1.  **Grab the Notebook:** Download or clone the `main.ipynb` file.
2.  **Find the Data:** The notebook looks for the data here:
    ```
    path = "/kaggle/input/creditcardfraud/creditcard.csv"
    ```
    **Important:** If you saved `creditcard.csv` somewhere else (like your local machine or Google Drive), **you need to change this `path` variable** in the notebook to point to the right spot!
3.  **Fire it Up:** Open the notebook in your favorite Jupyter environment and run the cells from top to bottom.

## The Workflow: What We Did

The notebook follows a pretty standard machine learning path:

1.  **Load Data:** Pulled the `creditcard.csv` into a pandas DataFrame.
2.  **Explore (EDA):** Did some initial digging:
    *   Looked at the first few rows (`head()`), data types (`info()`), and basic stats (`describe()`).
    *   Confirmed no missing values (`isnull().sum()`).
    *   Checked the `Class` distribution (`value_counts()`) and plotted it – really highlighted that 0.17% fraud rate!
3.  **Prep the Data:**
    *   **Features vs. Target:** Separated our input features (X) from the target variable (y, the `Class`). We dropped the `Time` column here, maybe because it's not strongly predictive on its own.
    *   **Train/Test Split:** Split the data: 70% for training, 30% for testing. Used `stratify=y` – this is crucial to make sure both the training and testing sets have the same tiny percentage of fraud cases. `random_state=42` keeps the split the same if we run it again.
    *   **Scaling (for KNN):** K-Nearest Neighbors cares about distances between data points. If features have wildly different ranges (like `Amount` vs. the `V` features), the ones with bigger values can dominate. So, we used `StandardScaler` to put all features on a similar scale. **Important:** We `fit` the scaler *only* on the training data (to avoid peeking at the test set) and then `transform` both train and test data. Decision Trees don't usually need this scaling.
4.  **Train & Evaluate Models:**
    *   **Decision Tree First:** Trained a `DecisionTreeClassifier` (using `entropy`) on the original (unscaled) data as a baseline.
    *   **Then KNN:**
        *   Trained a `KNeighborsClassifier` on the *scaled* data.
        *   Started by checking K=1, just like in the original notebook example.
        *   **Finding the "Best" K (Elbow Method):** K=1 isn't always optimal. We tried values of K from 1 to 39, calculated the error rate for each, and plotted it. The idea is to find the "elbow" point – where the error rate stops dropping sharply. This often gives a good balance. The plot suggested **K around 3** looked promising.
5.  **Judging Performance:** With imbalance, accuracy can be super misleading (a model predicting "not fraud" all the time would be 99.83% accurate!). So, we focused on:
    *   **Confusion Matrix:** A table showing correct/incorrect predictions for both classes (True Positives, True Negatives, False Positives, False Negatives).
    *   **Classification Report:** Gives us the key metrics per class:
        *   **Precision:** Out of all predicted frauds, how many were *actually* fraud? (High precision = fewer false alarms bothering legitimate customers).
        *   **Recall (Sensitivity):** Out of all *actual* frauds, how many did we catch? (High recall = catching more fraud).
        *   **F1-Score:** A combined score balancing Precision and Recall. Useful when both are important.
    *   We calculated overall accuracy too, but took it with a grain of salt.

## How Did It Go? (Results)

*   **Decision Tree:** Got that high ~99.9% accuracy, but looking closer at the fraud class (1):
    *   Precision: ~0.76 (About 76% of transactions it flagged as fraud were *actually* fraud).
    *   Recall: ~0.74 (It caught about 74% of the *actual* fraud cases).
    *   F1-Score: ~0.75
*   **KNN (K=1, Scaled Data):** With K=1, KNN seemed to do a bit better on precision for the fraud class in this specific run:
    *   Precision: ~0.86
    *   Recall: ~0.76
    *   F1-Score: ~0.81
*   **KNN (Optimal K ≈ 3):** The Elbow Method pointed towards K=3 as likely being a better, more stable choice than K=1. While the notebook showed detailed results for K=1, using K=3 would probably give a good balance between catching fraud (Recall) and not flagging too many good transactions (Precision).

## Key Takeaways & Next Steps

*   **Imbalance is King:** Dealing with the tiny percentage of fraud cases was the biggest factor driving our approach and how we evaluated the models.
*   **Prep Matters:** Stratified splitting is a must. Scaling features was vital for KNN's performance.
*   **KNN Looks Promising (with Tuning):** KNN, especially after finding a good K (like K=3 via the Elbow method), showed it could be effective.
*   **Metrics are Crucial:** Don't rely on accuracy alone! Precision, Recall, F1, and the confusion matrix tell a much more complete story for imbalanced problems.

**What could we try next?**

*   **Run KNN with K=3:** Actually train and evaluate the KNN model using the optimal K=3 found by the elbow method to see its specific scores.
*   **Tackle Imbalance Directly:** Explore techniques like:
    *   **SMOTE:** Creating synthetic fraud examples to balance the training data.
    *   **Undersampling:** Removing some of the non-fraud examples from the training data.
    *   Using models designed for imbalance (like `BalancedRandomForestClassifier`).
*   **Try Other Models:** Experiment with Logistic Regression, SVMs, Random Forests, Gradient Boosting (like XGBoost or LightGBM), and tune their settings.
*   **Feature Deep Dive:** Although the `V` features are anonymous, maybe analyze the `Amount` feature more, or see if any feature engineering ideas pop up.

