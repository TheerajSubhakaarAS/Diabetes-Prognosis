# Diabetes-Prognosis

This code uses the PIMA Diabetes dataset to perform data analysis, data preprocessing, model training, and prediction.

## Dependencies
The following dependencies are imported in this code:

* pandas (as pd)
* StandardScaler from sklearn.preprocessing
* train_test_split from sklearn.model_selection
* svm from sklearn
* accuracy_score from sklearn.metrics
* matplotlib.pyplot (as plt)
* seaborn (as sns)
* KNeighborsClassifier from sklearn.neighbors
* DecisionTreeClassifier from sklearn.tree
* numpy (as np)
* pickle
## Data Collection and Analysis
The PIMA Diabetes dataset is loaded into a pandas DataFrame using pd.read_csv(). The first 5 rows of the dataset are printed using the head() method, and the number of rows and columns in the dataset is printed using the shape attribute. The statistical measures of the dataset are obtained using the describe() method. The count of non-diabetic and diabetic patients in the dataset is obtained using value_counts(), and a pie chart is plotted to visualize the count of each category. The mean values of the attributes for non-diabetic and diabetic patients are obtained using groupby().

## Data Standardization
The data is standardized using StandardScaler from sklearn.preprocessing. The fit() method is used to fit the scaler object to the data, and the transform() method is used to transform the data.

## Train Test Split
The standardized data is split into training and testing sets using train_test_split from sklearn.model_selection. The test size is set to 0.2, stratify is set to Y, and random_state is set to 2.

## Model Training
Three models are trained on the data: SVM, K-nearest neighbor (KNN), and Decision Tree. The SVM classifier is trained using svm.SVC(), KNN classifier is trained using KNeighborsClassifier(), and the Decision Tree classifier is trained using DecisionTreeClassifier().

## Model Evaluation
The accuracy scores for each model are obtained using accuracy_score from sklearn.metrics. The accuracy scores are obtained separately for the training and testing data for each model.

## Making a Predictive System
An input data point is provided to the model, and the model predicts whether the person is diabetic or not. The input data is first standardized, then the model is used to make a prediction.

## Saving Trained Model
The trained models are saved using the pickle module. The SVM model is saved as diabetes_model_SVM.sav, the KNN model is saved as diabetes_model_KNN.sav, and the Decision Tree model is saved as diabetes_model_DT.sav.
