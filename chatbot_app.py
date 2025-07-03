# === FILE: chatbot_app.py ===
# Jadwal Piket Kelas dengan Streamlit

import torch
import torch.nn as nn
import numpy as np
import streamlit as st
import pandas as pd

# Fungsi tokenize tanpa NLTK
def tokenize(sentence):
    return sentence.lower().split()

# Fungsi stem sederhana (manual lowercase saja, bisa tambah regex bila perlu)
def stem(word):
    return word.lower()

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# Load model
data = torch.load("data.pth")

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Load dataset jadwal piket kelas
jadwal_df = pd.read_csv("dataset_jadwal_piket_kelas.csv")

# Response generator berdasarkan intent dan entitas
def get_response(intent, entity):
    if intent == "piket_hari_ini":
        hari_ini = pd.Timestamp.today().day_name()
        result = jadwal_df[jadwal_df['hari'].str.lower() == hari_ini.lower()]
    elif intent == "piket_hari_tertentu" and entity.get("hari"):
        result = jadwal_df[jadwal_df['hari'].str.lower() == entity["hari"].lower()]
    elif intent == "piket_berdasarkan_kelas" and entity.get("kelas"):
        result = jadwal_df[jadwal_df['kelas'].str.lower() == entity["kelas"].lower()]
    elif intent == "piket_berdasarkan_nama" and entity.get("nama"):
        result = jadwal_df[jadwal_df['nama'].str.lower().str.contains(entity["nama"].lower())]
    else:
        return "Maaf, saya belum dapat menemukan informasi piket yang sesuai."

    if not result.empty:
        list_piket = result.apply(
            lambda row: f"{row['hari']}: {row['nama']} dari kelas {row['kelas']} (Tugas: {row['tugas']})",
            axis=1
        ).tolist()
        return "Berikut jadwal piketnya:\n- " + "\n- ".join(list_piket)
    else:
        return "Tidak ditemukan jadwal piket yang sesuai."

# Ekstrak entitas manual dari teks
def extract_entity(text):
    hari_list = ["senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu"]
    entity = {}

    for h in hari_list:
        if h in text.lower():
            entity["hari"] = h
            break

    kelas_list = jadwal_df["kelas"].dropna().unique()
    for k in kelas_list:
        if k.lower() in text.lower():
            entity["kelas"] = k
            break

    nama_list = jadwal_df["nama"].dropna().unique()
    for n in nama_list:
        if n.lower().split()[0] in text.lower():
            entity["nama"] = n
            break

    return entity

# Prediksi intent
def predict_class(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).float().unsqueeze(0)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.40:
        return tag
    else:
        return "unknown"

# Streamlit App
st.title("🧹 Chatbot Jadwal Piket Kelas Semester Ganjil 2025/2026")
st.markdown("Tanyakan tentang jadwal piket berdasarkan hari, nama siswa, atau kelas.")

user_input = st.text_input("Ketik pertanyaan kamu:", "Siapa yang piket hari Jumat?")

if st.button("Tanya"):
    intent = predict_class(user_input)
    entity = extract_entity(user_input)
    response = get_response(intent, entity)
    st.write(response)
