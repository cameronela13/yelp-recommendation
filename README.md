# yelp-recommendation
Using Spark and a random forest model, I built a model-based recommendation system for a large Yelp dataset. The dataset includes information about Yelp users, businesses, and train and validation sets connecting users and businesses with star ratings. While I originally considered a hybrid recommendation system, item-based collaborative filtering only increased the final reported root-mean-squared error. Efficient parallel processing with Spark reduced execution time for this project, which ran for just over 90 seconds using all cores on my Intel Mac. If I were to do this project again, I could create more features using different aggregates or even features of higher degrees. However, additional features may not be a reliable answer, considering additional confounding and overfitting.

I used the PySpark API to import several large datasets as resilient distributed datasets (RDDs). Using mapping and grouping operations, I built features out of Yelp user and restaurant characteristics before concatenating them. Before training a final XGBRegressor model, I used 5-fold cross-validation to determine the optimal random forest hyperparameters. My goal was to reduce overfitting, so I cross-validated on max tree depth, number of trees, column sampling, and the L1 regularization parameter (alpha). After determining the best hyperparameters with an extensive grid search, I trained the XGBRegressor on the full training set and subsequently made predictions on an unseen validation set. The execution time, from reading in data to outputting validation set predictions, is printed at the end. Alongside the print time is the validation set distribution of errors and the validation set root-mean-squared error. I will include those metrics from my execution below:

Error Distribution:
>=0 and <1: 102083
>=1 and <2: 33007
>=2 and <3: 6139
>=3 and <4: 815
>=4: 0

RMSE:
0.9792830344697815

Execution Time:
96s

Executing the project:
This project is executed on a single Python file. It depends on the following non-built-in libraries: numpy, xgboost, and sklearn (currently, the only part using sklearn is commented out as it was only used for cross-validation). The code can be executed from the CLI using python3 in the following fashion, assuming you execute it from the directory where the Python file is located:
python3 ./yelp_recommendation.py <folder_path> <test_file_name> <output_file_name>
<folder_path>: The local path to a folder containing all input data used for feature engineering, cross-validation, and training. The train file in this folder is a CSV with Yelp user ID, business ID, and the Yelp user's rating of the business
<test_file_name>: The local path to the CSV containing the validation set with the same structure as the train set (the validation set is included in the aforementioned folder for simplicity, but its local path must still be provided)
<output_file_name>: The local path to the CSV file you choose to output data to
