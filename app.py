import streamlit as st
import torch
import torch.nn as nn
import pickle
import pretty_midi
import io

# ---- Model definition (must match training exactly) ----
class MelodyLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_layers=2):
        super(MelodyLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_output = lstm_out[:, -1, :]
        logits = self.fc(last_output)
        return logits

# ---- Load model and vocab (cached so it only loads once) ----
@st.cache_resource
def load_model():
    with open("vocab.pkl", "rb") as f:
        vocab_data = pickle.load(f)

    model = MelodyLSTM(vocab_data["vocab_size"])
    model.load_state_dict(torch.load("melody_lstm.pth", map_location="cpu"))
    model.eval()

    return model, vocab_data["pitch_to_token"], vocab_data["token_to_pitch"]

model, pitch_to_token, token_to_pitch = load_model()

# ---- Generation function ----
def generate_melody(model, seed_sequence, length=100, temperature=1.0):
    generated = list(seed_sequence)
    with torch.no_grad():
        for _ in range(length):
            input_seq = torch.tensor([generated[-50:]], dtype=torch.long)
            logits = model(input_seq)
            probabilities = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, 1).item()
            generated.append(next_token)
    return generated

def tokens_to_midi_bytes(tokens, token_to_pitch, note_duration=0.3):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    start_time = 0.0
    for token in tokens:
        pitch = token_to_pitch[token]
        note = pretty_midi.Note(velocity=80, pitch=pitch, start=start_time, end=start_time + note_duration)
        instrument.notes.append(note)
        start_time += note_duration
    midi.instruments.append(instrument)

    buffer = io.BytesIO()
    midi.write(buffer)
    buffer.seek(0)
    return buffer

# ---- Streamlit UI ----
st.title("🎵 Neural Melody Generator")
st.write("An LSTM-based AI model that generates original melodies, trained on the Lakh MIDI dataset.")

temperature = st.slider("Creativity (temperature)", min_value=0.5, max_value=1.5, value=1.0, step=0.1,
                          help="Lower = safer/more repetitive. Higher = more random/creative.")

length = st.slider("Melody length (notes)", min_value=20, max_value=200, value=100, step=10)

if st.button("🎼 Generate Melody"):
    import random
    # pick a random starting seed from vocab (simple starting point)
    seed = [random.choice(list(token_to_pitch.keys())) for _ in range(50)]

    with st.spinner("Composing..."):
        generated_tokens = generate_melody(model, seed, length=length, temperature=temperature)
        midi_buffer = tokens_to_midi_bytes(generated_tokens, token_to_pitch)

    st.success("Melody generated!")
    st.download_button(
        label="⬇️ Download MIDI file",
        data=midi_buffer,
        file_name="generated_melody.mid",
        mime="audio/midi"
    )
    st.write("Note: MIDI files don't play directly in browsers — download and open with a media player, or import into MuseScore/GarageBand.")