import streamlit as st
import tensorflow as tf
import requests
import os
import numpy as np

# --- Configuration ---
MODEL_URL = "https://github.com/constantinbender51-cmyk/Models/raw/refs/heads/main/spell_corrector.keras"
MODEL_PATH = "spell_corrector.keras"

st.set_page_config(page_title="Spell Corrector", page_icon="✨")

# --- Helper Functions ---

def download_model(url, save_path):
    """Downloads the model file if it doesn't exist."""
    if not os.path.exists(save_path):
        with st.spinner(f"Downloading model from {url}..."):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("Model downloaded successfully!")
            except Exception as e:
                st.error(f"Error downloading model: {e}")
                st.stop()
    return True

@st.cache_resource
def load_keras_model(path):
    """Loads the Keras model into memory."""
    try:
        # Note: If your model uses custom layers, pass them in the custom_objects dict
        # e.g., load_model(path, custom_objects={'CustomLayer': CustomLayer})
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_text(text):
    """
    PRE-PROCESSING PLACEHOLDER
    
    IMPORTANT: A raw .keras file usually requires specific tokenization 
    (converting text to numbers) that matches how it was trained.
    
    Since the tokenizer wasn't provided in the link, you must implement 
    the specific logic here (e.g., loading a pickle tokenizer, character mapping, etc.).
    """
    # Example logic (You likely need to replace this with your actual Tokenizer):
    # tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
    # sequence = tokenizer.texts_to_sequences([text])
    # padded = pad_sequences(sequence, maxlen=...)
    # return padded
    
    st.warning("⚠️ Tokenizer logic missing. Using dummy conversion (Text -> Bytes). Check `preprocess_text` in code.")
    return text 

def decode_output(prediction):
    """
    POST-PROCESSING PLACEHOLDER
    
    Convert the model's numeric output back to text.
    """
    # Example logic:
    # predicted_ids = np.argmax(prediction, axis=-1)
    # return tokenizer.sequences_to_texts(predicted_ids)[0]
    
    return str(prediction)

# --- Main UI ---

st.title("✨ AI Spell Corrector")
st.markdown("This app loads a custom Keras model to correct spelling errors.")

# 1. Download Model
download_model(MODEL_URL, MODEL_PATH)

# 2. Load Model
model = load_keras_model(MODEL_PATH)

if model:
    # 3. Input UI
    input_text = st.text_area("Enter text to correct:", placeholder="Type heere...")

    if st.button("Correct Spelling"):
        if input_text:
            # 4. Inference
            try:
                # A. Preprocess
                processed_input = preprocess_text(input_text)
                
                # B. Predict (logic depends heavily on model input shape)
                # If the model expects a string directly (rare), use:
                # prediction = model.predict([input_text])
                
                # If the model expects tensors (common), ensure processed_input is correct format
                # For now, we wrap it in a try-catch because we don't know the model's signature
                st.info("Running prediction...")
                
                # NOTE: This line will fail if preprocess_text doesn't return the exact tensor shape 
                # the model expects. 
                prediction = model.predict([processed_input]) 

                # C. Decode
                result = decode_output(prediction)

                # 5. Output UI
                st.subheader("Correction:")
                st.success(result)
                
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.markdown("""
                **Debugging Tip:** Keras models require specific input shapes (e.g., shape=(1, 50)). 
                You need to update the `preprocess_text` function in `app.py` to convert your 
                string input into the exact tensor format your model was trained on.
                """)
        else:
            st.warning("Please enter some text first.")
else:
    st.error("Could not load the model.")

# --- Footer ---
st.markdown("---")
st.caption("Deployed via Railway")