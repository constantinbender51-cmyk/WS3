import streamlit as st
import tensorflow as tf
import requests
import os
import numpy as np
import string

# --- Configuration ---
MODEL_URL = "https://github.com/constantinbender51-cmyk/Models/raw/refs/heads/main/spell_corrector.keras"
MODEL_PATH = "spell_corrector.keras"
LATENT_DIM = 128  # Matches the training script

st.set_page_config(page_title="Spell Corrector", page_icon="✨")

# --- Vocabulary Setup (Matches Training Script) ---
characters = sorted(list(string.ascii_lowercase))

# Encoder: a-z (1-26)
input_token_index = {char: i+1 for i, char in enumerate(characters)}

# Decoder: \t=1, \n=2, a-z=3+
target_token_index = {'\t': 1, '\n': 2}
for i, char in enumerate(characters):
    target_token_index[char] = i + 3

reverse_target_char_index = {i: char for char, i in target_token_index.items()}

# --- Helper Functions ---

def download_model(url, save_path):
    """Downloads the model if not present."""
    if not os.path.exists(save_path):
        with st.spinner(f"Downloading model..."):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("Model downloaded!")
            except Exception as e:
                st.error(f"Download failed: {e}")
                st.stop()

@st.cache_resource
def load_inference_models(path):
    """
    Loads the trained model and dissects it to create 
    separate Encoder and Decoder models for step-by-step inference.
    """
    try:
        # 1. Load the full training model
        # The file contains the architecture: [Enc_In, Dec_In] -> Dec_Out
        train_model = tf.keras.models.load_model(path)
        
        # 2. Extract Layers by Type/Order
        # Since layers weren't named in training, we rely on the order they were defined:
        # 1. Embedding (Encoder)
        # 2. GRU (Encoder)
        # 3. Embedding (Decoder)
        # 4. GRU (Decoder)
        # 5. Dense (Output)
        
        layers = [l for l in train_model.layers if len(l.weights) > 0 or isinstance(l, tf.keras.layers.GRU)]
        
        # Filter layers
        embeddings = [l for l in train_model.layers if isinstance(l, tf.keras.layers.Embedding)]
        grus = [l for l in train_model.layers if isinstance(l, tf.keras.layers.GRU)]
        denses = [l for l in train_model.layers if isinstance(l, tf.keras.layers.Dense)]
        
        if len(embeddings) < 2 or len(grus) < 2 or len(denses) < 1:
            st.error("Model structure mismatch: Could not find all required layers.")
            return None, None

        enc_emb_layer = embeddings[0]
        enc_gru_layer = grus[0]
        dec_emb_layer = embeddings[1]
        dec_gru_layer = grus[1]
        dec_dense_layer = denses[0]

        # 3. Reconstruct Encoder Model
        # Input -> Embedding -> GRU -> State
        encoder_inputs = tf.keras.Input(shape=(None,), name="inf_enc_in")
        enc_emb = enc_emb_layer(encoder_inputs)
        _, state_h = enc_gru_layer(enc_emb)
        encoder_model = tf.keras.Model(encoder_inputs, state_h)

        # 4. Reconstruct Decoder Model
        # Input + State -> Embedding -> GRU -> Dense -> Output + New State
        decoder_inputs = tf.keras.Input(shape=(None,), name="inf_dec_in")
        decoder_state_input_h = tf.keras.Input(shape=(LATENT_DIM,), name="inf_dec_state_in")
        
        dec_emb = dec_emb_layer(decoder_inputs)
        # Note: We must pass initial_state to the GRU for inference
        dec_outputs, state_h_new = dec_gru_layer(dec_emb, initial_state=decoder_state_input_h)
        dec_outputs = dec_dense_layer(dec_outputs)
        
        decoder_model = tf.keras.Model(
            [decoder_inputs, decoder_state_input_h],
            [dec_outputs, state_h_new]
        )

        return encoder_model, decoder_model

    except Exception as e:
        st.error(f"Error rebuilding models: {e}")
        return None, None

def decode_sequence(input_text, encoder_model, decoder_model):
    # 1. Preprocess Input
    input_text = input_text.lower()
    input_seq = np.zeros((1, len(input_text)), dtype="float32")
    
    for t, char in enumerate(input_text):
        if char in input_token_index:
            input_seq[0, t] = input_token_index[char]
            
    # 2. Encode
    states_value = encoder_model.predict(input_seq, verbose=0)

    # 3. Setup Decoder Loop
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_token_index['\t'] # Start char

    stop_condition = False
    decoded_sentence = ""
    max_len = 50
    
    while not stop_condition:
        output_tokens, h = decoder_model.predict([target_seq, states_value], verbose=0)

        # Sample token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        
        if sampled_token_index in reverse_target_char_index:
            sampled_char = reverse_target_char_index[sampled_token_index]
        else:
            sampled_char = ''
            
        if sampled_char == '\n' or len(decoded_sentence) > max_len:
            stop_condition = True
        else:
            decoded_sentence += sampled_char

        # Update
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = h

    return decoded_sentence

# --- Main UI ---

st.title("✨ Spell Corrector")
st.caption("Standard Seq2Seq GRU (No Attention)")

download_model(MODEL_URL, MODEL_PATH)

# Load and Reconstruct
enc_model, dec_model = load_inference_models(MODEL_PATH)

if enc_model and dec_model:
    text_input = st.text_input("Enter misspelled text:", value="compputer")
    
    if st.button("Correct"):
        if text_input:
            with st.spinner("Correcting..."):
                result = decode_sequence(text_input, enc_model, dec_model)
            st.success(f"Correction: **{result}**")
        else:
            st.warning("Please type something.")
else:
    st.error("Failed to load models.")