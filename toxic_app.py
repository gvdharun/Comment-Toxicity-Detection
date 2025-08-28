import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load your pre-trained CNN model
model = load_model('models/toxicity_cnn_model.h5')

# Toxicity classes
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Parameters (must match training)
MAX_SEQUENCE_LENGTH = 200
VOCAB_SIZE = 10000

# Dummy tokenizer: replace with your actual tokenizer logic
def dummy_tokenizer(text):
    # For demonstration only: generate random token indices
    return np.random.randint(1, VOCAB_SIZE, size=(MAX_SEQUENCE_LENGTH,))

def preprocess_text(text):
    tokens = dummy_tokenizer(text)
    tokens = np.expand_dims(tokens, axis=0)  # batch dim
    return tokens

def predict_single_comment(comment, threshold=0.5):
    x = preprocess_text(comment)
    preds = model.predict(x)[0]
    results = {}
    for label, prob in zip(labels, preds):
        pred_label = "Yes" if prob >= threshold else "No"
        results[label] = {"Prediction": pred_label, "Confidence": prob}
    return results

# Streamlit UI
st.title("Toxic Comment Detection App")

st.header("Real-time Single Comment Prediction")
user_comment = st.text_area("Enter comment text for toxicity prediction:")

if st.button("Predict"):
    if user_comment.strip():
        preds = predict_single_comment(user_comment)
        st.write("## Predictions:")
        for label, res in preds.items():
            st.write(f"- **{label.capitalize()}**: {res['Prediction']} (Confidence: {res['Confidence']:.4f})")
    else:
        st.warning("Please enter a comment to analyze.")

st.header("Sample Test Cases and Data Insights")

if st.checkbox("Show Sample Test Cases"):
    # Load sample data if available, else dummy
    try:
        sample_df = pd.read_csv('sample_test.csv')  # Replace with your actual file path
        st.dataframe(sample_df.head(10))
    except Exception as e:
        st.error(f"Sample data not found or error loading: {e}")

st.header("Model Performance Metrics")
st.markdown("""
- Training Accuracy: ~99.4%  
- Validation Accuracy: ~99.4%  
- Training Loss: ~0.13  
- Note: For detailed evaluation metrics, please refer to training reports.
""")

st.header("Bulk Predictions from CSV Upload")
uploaded_file = st.file_uploader("Upload CSV file with a 'comment_text' column", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        if 'comment_text' not in data.columns:
            st.error("CSV must contain a 'comment_text' column.")
        else:
            comments = data['comment_text'].astype(str).tolist()
            processed_tokens = np.array([dummy_tokenizer(txt) for txt in comments])
            preds = model.predict(processed_tokens)
            preds_df = pd.DataFrame(preds, columns=labels)
            results = pd.concat([data, preds_df], axis=1)
            st.write("### Predictions:")
            st.dataframe(results)

            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name='toxicity_predictions.csv',
                mime='text/csv',
            )
    except Exception as e:
        st.error(f"Error processing file: {e}")
