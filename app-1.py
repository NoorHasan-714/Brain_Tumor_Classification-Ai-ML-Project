import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
try:
    df = pd.read_csv('Brain_Tumor.csv')
    df = df.drop('Image', axis=1)
except FileNotFoundError:
    st.error("Error: 'Brain_Tumor.csv' not found. Please make sure the file is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the dataset: {e}")
    st.stop()

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model
def create_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
input_dim = X_train_scaled.shape[1]
model = create_model(input_dim)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,  # Increased epochs for potential better learning
    validation_split=0.2,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0  # Reduced verbosity for Streamlit app
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Streamlit App
st.title("Brain Tumor Detection")
st.write("This app uses a neural network to predict the presence of a brain tumor based on medical features.")
st.write("---")

st.subheader("Model Performance on Test Set")
st.write(f"**Test Loss:** {loss:.4f}")
st.write(f"**Test Accuracy:** {accuracy:.4f}")

st.subheader("Classification Report")
st.text(report)

st.subheader("Confusion Matrix")
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
ax_cm.set_xlabel('Predicted')
ax_cm.set_ylabel('Actual')
st.pyplot(fig_cm)

st.write("---")
st.subheader("Make a New Prediction")
st.write("Enter the feature values for a new patient:")

# Create input fields for each feature
feature_names = X.columns
new_features = {}
for feature in feature_names:
    new_features[feature] = st.number_input(f"{feature}:", value=np.mean(X[feature])) # Default to mean

if st.button("Predict"):
    new_data = pd.DataFrame([new_features])
    new_data_scaled = scaler.transform(new_data)
    prediction_proba = model.predict(new_data_scaled)[0][0]
    prediction = "Has Tumor" if prediction_proba > 0.5 else "No Tumor"
    confidence = f"{prediction_proba * 100:.2f}%"
    st.subheader("Prediction Result")
    st.write(f"**Predicted Class:** {prediction}")
    st.write(f"**Confidence:** {confidence}")

st.write("---")
st.subheader("Dataset Information")
if st.checkbox("Show Raw Data"):
    st.dataframe(df)

if st.checkbox("Show Class Distribution"):
    fig_class, ax_class = plt.subplots()
    sns.countplot(data=df, x='Class', ax=ax_class)
    ax_class.set_title('Class Distribution')
    ax_class.set_xticklabels(['No Tumor', 'Has Tumor'])
    st.pyplot(fig_class)

if st.checkbox("Show Feature Correlations"):
    fig_corr, ax_corr = plt.subplots()
    corr_matrix = df.drop('Class', axis=1).corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax_corr)
    ax_corr.set_title('Feature Correlation Matrix')
    st.pyplot(fig_corr)