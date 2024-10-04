import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

# Step 1: Load the data
df = pd.read_csv(r'C:\Users\Administrator\Desktop\ProjetosAIMIGUEL\ClothingRecomender\retail_data_crunched.csv')

# Step 2: Handle missing values
numeric_features = ['Transaction_ID', 'Customer_ID', 'Age', 'Total_Purchases', 'Amount', 'Total_Amount', 'Ratings']
categorical_features = ['Gender', 'Customer_Segment', 'Income', 'City', 'State', 'Country', 'Product_Category', 'Product_Brand', 'Product_Type', 'Shipping_Method', 'Payment_Method', 'Order_Status']

# Impute numeric features with median
numeric_imputer = SimpleImputer(strategy='median')
df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])

# Impute categorical features with mode
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])

# Step 3: Encode categorical features
# Label Encoding for binary/ordinal categories
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Customer_Segment'] = le.fit_transform(df['Customer_Segment'])

# Ordinal Encoding for 'Income'
income_encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
df['Income'] = income_encoder.fit_transform(df[['Income']])

# One-Hot Encoding for other categorical features
categorical_features_onehot = ['City', 'State', 'Country', 'Product_Category', 'Product_Brand', 'Product_Type', 'Shipping_Method', 'Payment_Method', 'Order_Status']
df_encoded = pd.get_dummies(df, columns=categorical_features_onehot)

# Step 4: Process date and time features
df_encoded['Date'] = pd.to_datetime(df_encoded['Date'])
df_encoded['DayOfWeek'] = df_encoded['Date'].dt.dayofweek
df_encoded['IsWeekend'] = df_encoded['DayOfWeek'].isin([5, 6]).astype(int)
df_encoded['Month'] = df_encoded['Date'].dt.month
df_encoded['Season'] = pd.cut(df_encoded['Date'].dt.month, bins=[0, 3, 6, 9, 12], labels=['Winter', 'Spring', 'Summer', 'Fall'])
df_encoded['Hour'] = pd.to_datetime(df_encoded['Time']).dt.hour
df_encoded['TimeOfDay'] = pd.cut(df_encoded['Hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])

# One-hot encode new categorical features
df_encoded = pd.get_dummies(df_encoded, columns=['DayOfWeek', 'Month', 'Season', 'TimeOfDay'])

# Step 5: Process text features (Feedback)
tfidf = TfidfVectorizer(max_features=100)  # Limit to top 100 features for simplicity
feedback_tfidf = tfidf.fit_transform(df_encoded['Feedback'].fillna(''))  # Fill NaN values
feedback_df = pd.DataFrame(feedback_tfidf.toarray(), columns=tfidf.get_feature_names_out())
df_encoded = pd.concat([df_encoded, feedback_df], axis=1)

# Step 6: Feature scaling
scaler = StandardScaler()
df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])

# Step 7: Create target variable
df_encoded['ProductID'] = df_encoded['products'].astype('category').cat.codes

# Step 8: Additional feature engineering
# Customer Lifetime Value (simplified as total amount spent)
clv = df_encoded.groupby('Customer_ID')['Total_Amount'].sum().reset_index()
clv.columns = ['Customer_ID', 'CLV']
df_encoded = df_encoded.merge(clv, on='Customer_ID', how='left')

# Purchase Frequency (number of transactions per customer)
purchase_freq = df_encoded.groupby('Customer_ID').size().reset_index(name='PurchaseFrequency')
df_encoded = df_encoded.merge(purchase_freq, on='Customer_ID', how='left')

# Product Popularity Score
product_popularity = df_encoded.groupby('products').size().reset_index(name='PopularityScore')
df_encoded = df_encoded.merge(product_popularity, on='products', how='left')

# Step 9: Drop unnecessary columns
columns_to_drop = ['Name', 'Email', 'Phone', 'Address', 'Zipcode', 'Date', 'Year', 'Time', 'Feedback', 'products']
df_encoded = df_encoded.drop(columns=columns_to_drop)

# Final step: Save the preprocessed data
df_encoded.to_csv('preprocessed_data.csv', index=False)

print("Data preprocessing completed. Preprocessed data saved as 'preprocessed_data.csv'")
print(f"Shape of preprocessed data: {df_encoded.shape}")
print("\nColumn names:")
print(df_encoded.columns.tolist())