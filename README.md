# Regression-analysis
Description of the task:
The wine industry shows a recent exponential growth as social drinking is on the rise. Nowadays, industry players are using product quality certifications to promote their products. This is a time-consuming process and requires the assessment given by human experts, which makes this process very expensive. Also, the price of wine depends on a rather abstract concept of wine appreciation by wine tasters, opinions among whom may have a high degree of variability.

Dataset contains wine samples to model wine quality based on physicochemical tests. The dataset contains a total of 12 variables, which were recorded for 4898 observations.

Input variables (based on physicochemical tests): 1 - fixed acidity 2 - volatile acidity 3 - citric acid 4 - residual sugar 5 - chlorides 6 - free sulfur dioxide 7 - total sulfur dioxide 8 - density 9 - pH 10 - sulphates 11 - alcohol .

Output variable (based on sensory data): 12 - quality (score between 0 and 10)

Task
Explore data (winequality.csv), check Null values.
Create a correlation analysis of independent variables against dependent variable, 'quality'.  Analyze how 'quality' values ​​are related with other variables using scatter plot. Show the contribution of each factor to the wine quality in the model.
Analyze the correlation between different variables. Show which features are more important in determining the wine quality.

Train a simple linear regression model (Ordinary Least Squares OLS).
Evaluate an LASSO regression model on the dataset. 
Compare model predicted results and choose a better model.

There is also set for predicting (without quality score) (pred_wine.csv). Use this set and chosen model set for predicting 'quality' values.
