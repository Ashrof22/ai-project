import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Define the URL for the CSV file on GitHub
file_url = "merged_ww_case_simplify.csv"

# Read the CSV file
try:
    df = pd.read_csv(file_url)

    # Convert 'Date' column to ordinal numbers
    df['Date_ordinal'] = pd.to_datetime(df['Date']).apply(lambda x: x.toordinal())

    # Assuming 'Date_ordinal' and 'case_nor_03d' are independent and dependent variables respectively
    x = df[['Date_ordinal']]  # Independent variable(s), adjust as needed
    y = df['case_nor_03d']  # Dependent variable, adjust as needed

    # Transforming the features to polynomial features
    polynomial_features = PolynomialFeatures(degree=2)  # You can change the degree as needed
    x_poly = polynomial_features.fit_transform(x)

    # Polynomial regression model
    model = LinearRegression()
    model.fit(x_poly, y)

    # Predictions
    y_pred = model.predict(x_poly)

    # Print the coefficients
    print("Coefficients:", model.coef_)

    # Combine actual and predicted values with all columns
    output_df = pd.concat([df, pd.DataFrame({'Predicted': y_pred})], axis=1)

    # Print the DataFrame
    print("Output DataFrame:")
    print(output_df)

except Exception as e:
    print("An error occurred:", e)
