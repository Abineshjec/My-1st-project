# My-1st-project
It's my future sale prediction code
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

# Load your sales data
# Replace 'sales_data.csv' with your dataset's file path
data = pd.read_csv('sales_data.csv')
# Assuming you have a date column, parse it as a datetime object
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Check for stationarity
def test_stationarity(timeseries):
    # Calculate rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()

    # Plot rolling statistics
    plt.figure(figsize=(12, 6))
    plt.plot(timeseries, label='Original')
    plt.plot(rolling_mean, label='Rolling Mean')
    plt.plot(rolling_std, label='Rolling Std')
    plt.legend()
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test
    result = adfuller(timeseries, autolag='AIC')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    if result[1] <= 0.05:
        print('Stationary (Reject Null Hypothesis)')
    else:
        print('Non-Stationary (Fail to Reject Null Hypothesis)')

# Make the time series stationary
data_diff = data['sales'].diff().dropna()
test_stationarity(data_diff)

# Plot ACF and PACF to determine p and q values for ARIMA
plot_acf(data_diff, lags=30)
plot_pacf(data_diff, lags=30)
plt.show()

# Fit ARIMA model
p = 1  # Replace with the appropriate lag value from PACF
d = 1  # Differencing order
q = 1  # Replace with the appropriate lag value from ACF

model = ARIMA(data['sales'], order=(p, d, q))
model_fit = model.fit(disp=0)

# Make predictions
forecast_period = 12  # Replace with the number of periods to forecast
forecast, stderr, conf_int = model_fit.forecast(steps=forecast_period)

# Create a date range for future predictions
future_dates = pd.date_range(start=data.index[-1], periods=forecast_period + 1, closed='right')

# Create a DataFrame for the forecast
forecast_df = pd.DataFrame({'forecast': forecast}, index=future_dates[1:])

# Plot original data and forecast
plt.figure(figsize=(12, 6))
plt.plot(data['sales'], label='Original Data')
plt.plot(forecast_df, label='Forecast')
plt.legend()
plt.title('Sales Forecast')
plt.show()

# Evaluate the model (optional)
# Split your data into training and testing sets and use them to evaluate the model's accuracy.
# Calculate Mean Squared Error (MSE) and other relevant metrics.
# You may also want to perform cross-validation for a more robust evaluation.

# Save the model (optional)
# If you're satisfied with your ARIMA model, you can save it for future use:
# model_fit.save('sales_arima_model.pkl')

# Load the model (for future predictions)
# To load the model for future sales predictions without training it again:
# loaded_model = ARIMAResults.load('sales_arima_model.pkl










FUTURE SALES PREDICTION




Predict of future sales promotion using AutoAI capabilities within IBM Watson Studio This tutorial guides you through training a model to predict the increase in sales of an item after promotion. In this tutorial, you will create an AutoAI experiment in IBM Watson Studio to build a model that analyzes your data and selects the best model type and estimators to produce, train, and optimize pipelines, which are model candidates. After reviewing the pipelines, you will save one as a model, deploy it, then test it to get a prediction.

Pre-requisites

IBM Cloud Account: Visit https://ibm.biz/autoailab and fill in your details to create an account or click 'Log in' if you already have an account.

Create instances of the following: Click on 'Catalog', look for Object Storage, give it a name (or leave the default) and click 'Create'. Do the same to create instances for Watson Machine Learning and Watson Studio.

Dataset: Download the Sales.csv dataset.

The dataset contains the follwing columns:

Class which describes the Product type Cost is the Unit price Promotion is the Index of amount spent on a particular promotion Before describes the Revenue before promotion After describes the Revenue after promotion Step 1: Build and train the model In your Watson Studio instance, click Get Started > Create a project > Create an empty project

Give your project a name and an optional description, connect the Object Storage instance created earlier and click 'Create'.

Once created, click on Add to project > AutoAI Experiment

Give your AutoAI experiment a name and an optional description, connect the Watson Machine Learning Service instance created earlier, leave the Compute Configuration as default and click 'Create'.

In the 'Add training data' section, drag and drop or browse for the Sales.csv file downloaded earlier to upload.

Next we're going to train the model.

Select 'Increase' as the column to predict. The prediction type recommended by AutoAI is Regression and the opetimized metric is RMSE. These can be changed by clicking on 'Configure prediction' but we're going to go with the recommended ones and click 'Run Experiment'.

As the model trains, you will see an infographic that shows the process of building the pipelines.

For a list of estimators available with each machine learning technique in AutoAI, you can check out: AutoAI implementation detail

Once the pipeline creation is complete, you can see all the ranked pipelines in a leaderboard. You can view details of each pipeline by clicking on the '>' and compare the pipelines by clicking on 'Compare pipelines'.

Choose 'Save as model' and then click 'Save' for thr pipeline ranked 1st. This saves the pipeline as a Machine Learning asset in your project so you can deploy, train, and test it.

Step 2: Deploy the trained model Before you can use your trained model to make predictions on new data, you must deploy the model.

The model can be deployed from the model details page. You can access the model details page in one of these ways:

Clicking on the model name in the notification displayed when you save the model...or, Open the Assets page for the project containing the model and click the model name in the Machine Learning Model section.

From the model details page, go to the 'Deployments' tab, click 'Add Deployment'.

Give your deployment a name, an optional description, select “Web service” as the Deployment type and click 'Save'.

Once saved, click on the deployment name to view the deployment details page.

Step 3: Test the deployed model The deployed model can be tested from the deployment details page. On the 'Test' tab of the deployment details page, test data can be entered in the fields provided or in JSON format.

Note that the test data replicates the data fields for the model with the exception of the prediction field.

Enter the following test data in the JSON editor (or alternatively enter the values into the fields):

{"input_data":[{ "fields": ["Class","Cost","Promotion","Before","After"], "values": [["Luxury",13.357,1920, 117440,125073]] }]} Click 'Predict' to predict the increase in sales for that item after the promotion.



