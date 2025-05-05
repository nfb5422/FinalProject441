import math
import random
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
from pydub import AudioSegment
from pydub.playback import play
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import os
import pyttsx3
import ollama
import logging
import warnings

# Suppress warnings for better readability
warnings.filterwarnings("ignore", message=".*tensor.storage.*")

# Define tour guide agents with different personalities
agents = {
    "1": {
        "name": "Historic Erie Guide",
        "system_prompt": "You are a knowledgeable and friendly tour guide for Erie, Pennsylvania. You specialize in local history and think step-by-step to explain notable landmarks and historical events in the city.",
        "temperature": 0.5,
        "max_tokens": 100
    },
    "2": {
        "name": "Food Enthusiast",
        "system_prompt": "You are a lively Erie local who loves food and local food spots in Erie, PA. You guide users through the best places to eat, drink, and enjoy local foodie culture based on the local food joints.",
        "temperature": 0.7,
        "max_tokens": 100
    },
    "3": {
        "name": "Nature and Outdoor Expert",
        "system_prompt": "You are an enthusiastic outdoor guide focused on Erie’s nature. You provide step-by-step tips for exploring parks, trails, Presque Isle State Park, and other outdoor destinations.",
        "temperature": 0.6,
        "max_tokens": 100
    }
}

# Function to initialize pyttsx3
def initialize_tts_engine():
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    return engine

# Function to generate a random ticket number (for potential future features)
def generate_ticket_number():
    return f"TICKET-{random.randint(100000, 999999)}"

# Load and chunk Erie guide file for RAG
def load_and_chunk_document(file_path, chunk_size=500, chunk_overlap=50):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError as e:
        print(f"Error decoding file {file_path}: {e}. Attempting with 'ignore' mode.")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()

    if not content.strip():
        print("Error: No content was found in the file. Ensure it contains readable text.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = [
        {"id": f"chunk_{i}", "text": chunk, "metadata": {"source": os.path.basename(file_path)}}
        for i, chunk in enumerate(text_splitter.split_text(content))
    ]
    return chunks

# Set up ChromaDB for RAG
def setup_chroma_db(chunks, collection_name="erie_knowledge"):
    client = chromadb.Client()
    try:
        client.delete_collection(collection_name)
    except:
        pass
    collection = client.create_collection(name=collection_name)
    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks]
    )
    return collection

# Retrieve relevant context from ChromaDB
def retrieve_context(collection, query, n_results=3):
    results = collection.query(query_texts=[query], n_results=n_results)
    return [doc for doc_list in results.get("documents", []) for doc in doc_list]

# Updated text-to-speech function using pyttsx3
def text_to_speech(text, tts_engine):
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print(f"Error during text-to-speech: {e}")

# Function to select an agent
def select_agent():
    print("Select a tour guide personality for your Erie visit:")
    for key, agent in agents.items():
        print(f"{key}. {agent['name']}")
    choice = input("Enter the number of your chosen guide: ")
    return agents.get(choice, agents["1"])

# Main chat loop
def main():
    tts_engine = initialize_tts_engine()

    guide_path = "Erie_guide.txt"  # Local tour guide data
    guide_chunks = load_and_chunk_document(guide_path)

    if not guide_chunks:
        print("Error: No content was found in the guide file.")
        return

    collection = setup_chroma_db(guide_chunks)

    selected_agent = select_agent()
    model = 'llama3.2'
    messages = [{'role': 'system', 'content': selected_agent['system_prompt']}]
    options = {
        'temperature': selected_agent['temperature'],
        'max_tokens': selected_agent['max_tokens']
    }

    print(f"You are now exploring Erie with: {selected_agent['name']}. Type '/exit' to leave.")

    while True:
        user_input = input("You: ")

        if user_input.strip().lower() == '/exit':
            print("Guide: Safe travels! Enjoy your time in Erie!")
            break

        context = retrieve_context(collection, user_input)
        context_text = "\n\n".join(context) if context else "No relevant information found in the guide."
        user_input += f"\n\nContext: {context_text}"

        messages.append({'role': 'user', 'content': user_input})
        response = ollama.chat(model=model, messages=messages, stream=False, options=options)

        print(f"Guide: {response.message.content}")
        text_to_speech(response.message.content, tts_engine)

if __name__ == "__main__":
    main()

