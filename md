# Import necessary libraries
import streamlit as st  # For creating the interactive web app
from sklearn.datasets import load_iris  # For loading the built-in Iris dataset
from sklearn.linear_model import LinearRegression  # For training a simple model
import numpy as np  # For numerical computations
import pandas as pd  # For working with tabular data

# Load and prepare the Iris dataset
def load_data():
    iris = load_iris()  # Load the dataset from scikit-learn
    X = iris.data  # Extract features (sepal/petal length and width)
    y = iris.target  # Extract target labels (0 = Setosa, 1 = Versicolor, 2 = Virginica)

    # Create a DataFrame for better data handling and display
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['species'] = y  # Add target labels as a new column

    return X, y, df  # Return features, target, and full dataset

# Train the machine learning model using Linear Regression
def train_model(X, y):
    model = LinearRegression()  # Create a Linear Regression model instance
    model.fit(X, y)  # Train the model with the data
    return model  # Return the trained model

# Predict the Iris species using the trained model and user input
def predict_species(model, sepal_length, sepal_width, petal_length, petal_width):
    # Format the inputs into a 2D array as required by scikit-learn
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Use the model to predict a numeric label (e.g., 0.6 ≈ 1)
    predicted_class = model.predict(input_data)

    # Map numeric prediction to actual species name
    species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    predicted_species = species_map.get(round(predicted_class[0]), 'Unknown')
    
    return predicted_species  # Return the predicted species name

# Define the main function to render the Streamlit app
def main():
    # Title and description at the top of the page
    st.markdown("<h1 style='text-align: center; color: yellow;'>Iris Species Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Enter the measurements below to predict the species of the Iris flower.</h2>", unsafe_allow_html=True)

    try:
        # Load the dataset and train the model
        X, y, df = load_data()
        model = train_model(X, y)

        # Input form for users: use columns for a clean layout
        col1, col2 = st.columns(2)
        with col1:
            sepal_length = st.text_input("Sepal Length (cm)", value="5.1")
            sepal_width = st.text_input("Sepal Width (cm)", value="3.5")
        with col2:
            petal_length = st.text_input("Petal Length (cm)", value="1.4")
            petal_width = st.text_input("Petal Width (cm)", value="0.2")

        # Placeholder where the prediction result will be displayed
        result_placeholder = st.empty()

        # When the user clicks the "Predict" button
        if st.button("Predict"):
            try:
                # Convert user inputs from strings to floats
                sepal_length = float(sepal_length)
                sepal_width = float(sepal_width)
                petal_length = float(petal_length)
                petal_width = float(petal_width)

                # Make the prediction
                predicted_species = predict_species(model, sepal_length, sepal_width, petal_length, petal_width)

                # Show the prediction result in a styled format
                result_placeholder.markdown(
                    f"<h3 style='text-align: center; color: green;'>Predicted Species: {predicted_species}</h3>",
                    unsafe_allow_html=True
                )

                # Show a toast-style notification
                st.toast(f"Prediction Complete: {predicted_species}", icon="🌱")

            except ValueError:
                st.error("Please enter valid numerical values for all inputs.")

        # Show the full dataset at the bottom
        st.subheader("Iris Dataset")
        st.dataframe(df)

    except Exception as e:
        # Display error if something goes wrong in the app
        st.error(f"An error occurred: {str(e)}")

# Execute the app only if this script is run directly
if __name__ == "__main__":
    main()
