# that  provide a user interface for the Lung cancer detection model.
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Load the dataset
dataset = pd.read_csv('dataset.csv')

# Display the dataset
st.write(dataset)

# Display the dataset shape
st.write(dataset.shape)

# Display the dataset columns
st.write(dataset.columns)

# Display the dataset description
st.write(dataset.describe())

# Display the dataset correlation
st.write(dataset.corr())

# Display the dataset correlation heatmap
st.write(sns.heatmap(dataset.corr()))

# Display the dataset correlation heatmap with annotations
st.write(sns.heatmap(dataset.corr(), annot=True))

# Display the dataset correlation heatmap with annotations and cmap
st.write(sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm'))