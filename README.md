## Description

The primary objective of this project is to forecast the outlook of survey participants by analyzing their responses within the framework of a specific course. It is essential to note that the project's aim is not for immediate application in real-world scenarios. Nevertheless, the methodologies and analytical techniques utilized in the development of the predictive models may offer valuable insights for relevant applications.

For those interested in the foundational research that informs this project, access to the original scholarly articles is provided [here](https://www.sciencedirect.com/science/article/pii/S2352340923004948). Additionally, the `docs` folder contains a direct link to these resources, alongside the original dataset and other pertinent materials.

The model training process encompasses a comprehensive series of steps, outlined as follows:

* *Data Preprocessing:* This phase includes handling missing values through imputation, employing Synthetic Minority Over-sampling Technique (SMOTE) for balancing the dataset, and selecting relevant features.

* *Train-test Set Division:* The dataset is divided into training and testing subsets, with the training set undergoing further partitioning via K-Fold Cross-Validation. The test set is reserved for subsequent evaluations of the model's performance.

* *Model Optimization:* Execution of a grid search algorithm to identify and select the optimal models based on predefined criteria.

* *Performance Evaluation:* The selected models are assessed using a suite of metrics, including Area Under the Curve (AUC), Classification Accuracy (CA), Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). The evaluation results, along with the confusion matrix of the top-performing model, are systematically documented in an Excel file for easy review and comparison.

* *Model Preservation and Utilization:* Once the models have been trained, they are preserved in .joblib format for future use. These saved models can be efficiently reloaded and evaluated against a specific segment of the original dataset that has been pre-arranged for this purpose.

This procedural framework ensures a thorough and systematic approach to model development and evaluation, aiming to foster an understanding of predictive modeling techniques within an academic context.


## Installation
```powershell
py -m venv venv

.\venv\Scripts\activate

pip install -r requirements.txt

py project-script.py

# Deactivate virtual environment when you are done
.\venv\Scripts\deactivate.bat
```

