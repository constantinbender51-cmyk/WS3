import streamlit as st
import tensorflow as tf
import requests
import os
import numpy as np
import string

# --- Configuration ---
MODEL_URL = "https://github.com/constantinbender51-cmyk/Models/raw/refs/heads/main/spell_corrector.keras"
MODEL_PATH = "spell_corrector.keras"
LATENT_DIM = 256 # Must match training script

st.set_page_config(page_title="Spell Corrector", page_icon="✨")

# --- Vocabulary Setup (Must match training script exactly) ---
characters = sorted(list(string.ascii_lowercase))

# Encoder Mappings
input_token_index = {char: i+1 for i, char in enumerate(characters)}
reverse_input_char_index = {i: char for char, i in input_token_index.items()}

# Decoder Mappings (Train script: \t=1, \n=2, chars=3+)
target_token_index = {'\t': 1, '\n': 2}
for i, char in enumerate(characters):
    target_token_index[char] = i + 3
reverse_target_char_index = {i: char for char, i in target_token_index.items()}

# --- Helper Functions ---

def download_model(url, save_path):
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
    Loads the training model and reconstructs the separate 
    Encoder and Decoder models needed for inference.
    """
    try:
        # 1. Load the full training model
        train_model = tf.keras.models.load_model(path)
        
        # 2. Extract Layers by Name or Type
        # We assume the layer naming from the training script holds
        try:
            encoder_gru = train_model.get_layer('encoder_gru')
            encoder_state_concat = train_model.get_layer('encoder_state_concat')
            decoder_gru = train_model.get_layer('decoder_gru')
            attention_layer = train_model.get_layer('attention_layer')
            output_dense = train_model.get_layer('output_dense')
            
            # Embeddings are usually auto-named 'embedding' and 'embedding_1'
            # We filter layers to find them to be safe
            embeddings = [l for l in train_model.layers if isinstance(l, tf.keras.layers.Embedding)]
            encoder_embedding_layer = embeddings[0] # First one defined in script
            decoder_embedding_layer = embeddings[1] # Second one defined in script
            
        except ValueError as e:
            st.error(f"Layer extraction failed. Did you change layer names? Error: {e}")
            return None, None

        # 3. Reconstruct Encoder Model
        # Input -> Embedding -> Bi-GRU -> [Outputs, Fwd_h, Bwd_h] -> Concat -> [Outputs, Combined_h]
        enc_inputs = tf.keras.Input(shape=(None,), name='inf_enc_in')
        enc_emb = encoder_embedding_layer(enc_inputs)
        enc_out, state_h_fwd, state_h_bwd = encoder_gru(enc_emb)
        state_h = encoder_state_concat([state_h_fwd, state_h_bwd])
        
        encoder_model = tf.keras.Model(enc_inputs, [enc_out, state_h])

        # 4. Reconstruct Decoder Model
        # We need to manually feed states and encoder outputs for Attention
        dec_inputs = tf.keras.Input(shape=(None,), name='inf_dec_in')
        dec_state_in = tf.keras.Input(shape=(LATENT_DIM * 2,), name='inf_dec_state_in')
        enc_output_in = tf.keras.Input(shape=(None, LATENT_DIM * 2), name='inf_enc_out_in')

        dec_emb = decoder_embedding_layer(dec_inputs)
        dec_out, dec_state_out = decoder_gru(dec_emb, initial_state=dec_state_in)
        
        # Attention: connects Decoder Out (query) + Encoder Out (value)
        context_vector = attention_layer([dec_out, enc_output_in])
        
        # Concat + Dense
        dec_concat = tf.keras.layers.Concatenate(axis=-1)([dec_out, context_vector])
        dec_final = output_dense(dec_concat)
        
        decoder_model = tf.keras.Model(
            [dec_inputs, dec_state_in, enc_output_in],
            [dec_final, dec_state_out]
        )

        return encoder_model, decoder_model

    except Exception as e:
        st.error(f"Error rebuilding models: {e}")
        return None, None

def decode_sequence(input_text, encoder_model, decoder_model):
    # 1. Preprocess Input
    input_text = input_text.lower()
    # Map chars to integers
    input_seq = np.zeros((1, len(input_text)), dtype="float32")
    for t, char in enumerate(input_text):
        if char in input_token_index:
            input_seq[0, t] = input_token_index[char]
            
    # 2. Encode
    enc_outs, states_value = encoder_model.predict(input_seq, verbose=0)

    # 3. Decode Loop
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_token_index['\t'] # Start token

    stop_condition = False
    decoded_sentence = ""
    max_len = 50 # Safety limit
    
    while not stop_condition:
        # Predict next char
        output_tokens, h = decoder_model.predict(
            [target_seq, states_value, enc_outs], 
            verbose=0
        )

        # Sample best token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        
        if sampled_token_index in reverse_target_char_index:
            sampled_char = reverse_target_char_index[sampled_token_index]
        else:
            sampled_char = ''
            
        # Exit conditions
        if sampled_char == '\n' or len(decoded_sentence) > max_len:
            stop_condition = True
        else:
            decoded_sentence += sampled_char

        # Update loop vars
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = h

    return decoded_sentence

# --- Main UI ---

st.title("✨ Seq2Seq Spell Corrector")
st.markdown("Enter a misspelled word (e.g., 'compputer') to correct it using the Attention model.")

download_model(MODEL_URL, MODEL_PATH)
enc_model, dec_model = load_inference_models(MODEL_PATH)

if enc_model and dec_model:
    # Form
    with st.form("correction_form"):
        text_input = st.text_input("Misspelled Word:", value="intellgence")
        submitted = st.form_submit_button("Correct")
        
        if submitted:
            if not text_input:
                st.warning("Please type a word.")
            else:
                with st.spinner("Correcting..."):
                    result = decode_sequence(text_input, enc_model, dec_model)
                
                st.subheader("Result:")
                st.success(result)
                
    st.markdown("---")
    st.markdown("### Debug Info")
    st.text(f"Vocab Size: {len(input_token_index)} chars")
    st.text(f"Model Latent Dim: {LATENT_DIM}")

else:
    st.error("Failed to initialize inference models.")