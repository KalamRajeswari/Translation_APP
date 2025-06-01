import tkinter as tk
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, LSTM, Dense

# -------------------- Load Dataset --------------------
df = pd.read_csv("french_tamil_words.csv")
french_words = df['french'].astype(str)
tamil_words = df['tamil'].astype(str)

# -------------------- Create Vocab --------------------
input_chars = sorted(set("".join(french_words)))
target_chars = sorted(set("".join(tamil_words)) | set(["\t", "\n"]))

input_token_index = {ch: i for i, ch in enumerate(input_chars)}
target_token_index = {ch: i for i, ch in enumerate(target_chars)}
reverse_target_index = {i: ch for ch, i in target_token_index.items()}

num_encoder_tokens = len(input_chars)
num_decoder_tokens = len(target_chars)
max_encoder_seq_length = 5
max_decoder_seq_length = max([len(txt) for txt in tamil_words]) + 2

latent_dim = 256

# -------------------- Load Model --------------------
model = load_model("french_to_tamil_model.h5")

# -------------------- Inference Models --------------------
# Encoder
encoder_inputs = model.input[0]
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder
decoder_inputs = model.input[1]
decoder_lstm = model.layers[3]
decoder_dense = model.layers[4]

# Decoder Inference
decoder_state_input_h = Input(shape=(latent_dim,), name="decoder_input_h")
decoder_state_input_c = Input(shape=(latent_dim,), name="decoder_input_c")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens), name="decoder_inputs_infer")

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# -------------------- Translator Functions --------------------
def encode_input(word):
    x = np.zeros((1, max_encoder_seq_length, num_encoder_tokens))
    for t, ch in enumerate(word):
        if ch in input_token_index:
            x[0, t, input_token_index[ch]] = 1.
    return x

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.

    decoded_sentence = ''
    for _ in range(max_decoder_seq_length):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_index[sampled_token_index]

        if sampled_char == '\n':
            break
        decoded_sentence += sampled_char

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1
        states_value = [h, c]

    return decoded_sentence

# -------------------- GUI --------------------
def translate():
    word = entry.get().strip().lower()
    if len(word) != 5:
        result_label.config(text="❌ Please enter exactly 5-letter French word", fg="red")
        return
    if any(ch not in input_token_index for ch in word):
        result_label.config(text="❌ Invalid characters in input", fg="red")
        return

    input_seq = encode_input(word)
    translated = decode_sequence(input_seq)
    result_label.config(text=f"✅ Tamil: {translated}", fg="green")

root = tk.Tk()
root.title("French ➜ Tamil Translator (5-letter only)")
root.geometry("450x220")

tk.Label(root, text="Enter a 5-letter French word:", font=("Arial", 13)).pack(pady=10)
entry = tk.Entry(root, font=("Arial", 14), width=30)
entry.pack()

tk.Button(root, text="Translate", font=("Arial", 12), command=translate).pack(pady=10)
result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.pack(pady=10)

root.mainloop()
