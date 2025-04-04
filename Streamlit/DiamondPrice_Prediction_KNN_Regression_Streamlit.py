import numpy as np
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.title(":red[Diamond Price Prediction]")

# Load dataset
df = pd.read_csv("diamonds.csv")

# Log transformation
df["log_carat"] = np.log(df["carat"])
df["log_price"] = np.log(df["price"])

# Drop original carat and price columns
df.drop(["carat", "price"], axis=1, inplace=True)

# Define ordinal rankings for categorical features
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']  # Worst to best
color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']  # Worst (J) to best (D)
clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']  # Worst to best

# Ordinal Encoding
encoder = OrdinalEncoder(categories=[cut_order, color_order, clarity_order])
df[['cut', 'color', 'clarity']] = encoder.fit_transform(df[['cut', 'color', 'clarity']])

# Splitting features and target
y = df.pop("log_price")
X = df

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Standard Scaling (critical for KNN)
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Model Training
regression = KNeighborsRegressor(n_neighbors=4)
regression.fit(X_train, y_train)

# Model Evaluation
y_pred_test = regression.predict(X_test)
y_pred_train = regression.predict(X_train)

mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

st.write(f"🔹 **Training Mean Squared Error (MSE):** {mse_train:.4f}")
st.write(f"🔹 **Training Accuracy (R² Score):** {r2_train:.4f}")

st.write(f"🔹 **Testing Mean Squared Error (MSE):** {mse_test:.4f}")
st.write(f"🔹 **Testing Accuracy (R² Score):** {r2_test:.4f}")

# Streamlit Input UI for Prediction
st.subheader("Enter Diamond Features for Prediction")

log_carat = np.log(st.number_input("Carat Weight", min_value=0.1, max_value=5.0, step=0.01))

cut_options = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
select_cut = st.selectbox("Diamond Cut", cut_options)

color_options = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
select_color = st.selectbox("Diamond Color", color_options)

clarity_options = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
select_clarity = st.selectbox("Diamond Clarity", clarity_options)

depth = st.number_input("Depth Percentage", min_value=50.0, max_value=80.0, step=0.1)
table = st.number_input("Table Percentage", min_value=50.0, max_value=80.0, step=0.1)

x_dim = st.number_input("X Dimension (mm)", min_value=0.0, max_value=10.0, step=0.1)
y_dim = st.number_input("Y Dimension (mm)", min_value=0.0, max_value=10.0, step=0.1)
z_dim = st.number_input("Z Dimension (mm)", min_value=0.0, max_value=10.0, step=0.1)

if st.button("Predict Diamond Price"):
    # Manually map user input to ordinal values
    cut_mapping = {v: i for i, v in enumerate(cut_order)}
    color_mapping = {v: i for i, v in enumerate(color_order)}
    clarity_mapping = {v: i for i, v in enumerate(clarity_order)}
    
    query_point = pd.DataFrame([{
        "log_carat": log_carat,
        "cut": cut_mapping[select_cut],
        "color": color_mapping[select_color],
        "clarity": clarity_mapping[select_clarity],
        "depth": depth,
        "table": table,
        "x": x_dim,
        "y": y_dim,
        "z": z_dim
    }])

    # Ensure column order matches training data
    query_point = query_point[X_train.columns]

    # Scale the input
    query_point_scaled = pd.DataFrame(scaler.transform(query_point), columns=X_train.columns)

    # Predict and convert log price back to USD
    log_price_prediction = regression.predict(query_point_scaled)[0]
    price_prediction = np.exp(log_price_prediction)

    st.success(f"Predicted Diamond Price: **${price_prediction:.2f}**")