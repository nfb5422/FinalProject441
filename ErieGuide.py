import math
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import os
import pyttsx3
import ollama


# Defines the different agents depending on th aspect of Erie you would like a guide to, history, food, or nature stops will be the focus for this project
agents = {
    "1": {
        "name": "Tom, Historic Erie Guide",
        "system_prompt": "You are a knowledgeable and friendly tour guide for Erie, Pennsylvania. You specialize in local history and think step-by-step to explain notable landmarks and historical events in the city. Your reccomendations should only consist of whats  relevant in the Erie_guide.txt whenever possible.  You should also list the coordinates everytime you introduce a new landmark or location, like **Presque Isle State Park (42.1115, -80.1513). Do not add any other symbols around the coordinates, make sure they follow the form  (42.1115, -80.1513)",
        "temperature": 0.1,
        "max_tokens": 100
    },
    "2": {
        "name": "Michelle, Food Enthusiast",
        "system_prompt": "You are a lively Erie local who loves food and local food spots in Erie, PA. You guide users through the best places to eat, drink, and enjoy local foodie culture based on the local food joints. Your reccomendations should only consist of whats relevant in the Erie_guide.txt whenever possible.  You should also list the coordinates everytime you introduce a new landmark or location, like **Presque Isle State Park (42.1115, -80.1513). Do not add any other symbols around the coordinates, make sure they follow the form  (42.1115, -80.1513)",
        "temperature": 0.1,
        "max_tokens": 100
    },
    "3": {
        "name": "Julie, Nature and Outdoor Expert",
        "system_prompt": "You are an enthusiastic outdoor guide focused on Erie’s nature. You provide step-by-step tips for exploring parks, trails, Presque Isle State Park, and other outdoor destinations. Your reccomendations should only consist of whats relevent in the Erie_guide.txt whenever possible. You should also list the coordinates everytime you introduce a new landmark or location, like **Presque Isle State Park (42.1115, -80.1513). Do not add any other symbols around the coordinates, make sure they follow the form  (42.1115, -80.1513)",
        "temperature": 0.1,
        "max_tokens": 100
    }
}

# Chnaged the engine to intialize a different voice for each agent, including Linda who I had to add via the settings
def initialize_tts_engine(agent_name):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)

    voices = engine.getProperty('voices')

    
    if "Tom" in agent_name:
        for voice in voices:
            if "David" in voice.name:
                engine.setProperty('voice', voice.id)
                break
    elif "Michelle" in agent_name:
        for voice in voices:
            if "Zira" in voice.name:
                engine.setProperty('voice', voice.id)
                break
    elif "Julie" in agent_name:
        for voice in voices:
            if "Linda" in voice.name:
                engine.setProperty('voice', voice.id)
                break

    return engine

# Loads and Chunks the document I stored with resources from the Internet, giving rough outline to what landmarks and locaions should be highlighted
# Based on the RAG lab implementation
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

# Sets up chroma_db to use in the responses called "erie knowledge"
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

# Gets what is important from the "erie knowledge"
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

# Basic Haverine function used to alculate the distance from the campus
def haversine_distance(lat1, lng1, lat2, lng2):

    # Radius of the Earth in kilometers
    R = 6371.0
    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lng1_rad = math.radians(lng1)
    lat2_rad = math.radians(lat2)
    lng2_rad = math.radians(lng2)
    # Difference in coordinates
    dlat = lat2_rad - lat1_rad
    dlng = lng2_rad - lng1_rad
    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    # Distance in kilometers
    return R * c

# Tool used to append the distance 
def append_distance_to_coordinates(text, behrend_coords):
    import re

    # Parses through to find the coordinates of the landmark
    coord_pattern = r'\(([-+]?\d{1,2}\.\d+),\s*([-+]?\d{1,3}\.\d+)\)'

    def repl(match):
        lat, lng = float(match.group(1)), float(match.group(2))
        dist_km = haversine_distance(behrend_coords[0], behrend_coords[1], lat, lng)
        dist_miles = dist_km * 0.621371
        return f"({lat}, {lng}) (~{dist_miles:.1f} miles from Behrend Campus)"

    return re.sub(coord_pattern, repl, text)


# Function that selects the agent used in the run throguh of the project
def select_guide_type():
    print("Select a tour guide personality for your Erie visit:")
    for key, agent in agents.items():
        print(f"{key}. {agent['name']}")
    choice = input("Enter the number of your chosen guide: ")
    return agents.get(choice, agents["1"])

# Main function
def main():
    

    guide_path = "C:/Users/labadmin/game 450/spring2025-labs/FinalProject441/Erie_guide.txt"  # Local tour guide data
    guide_chunks = load_and_chunk_document(guide_path)

    if not guide_chunks:
        print("Error: No content was found in the guide file.")
        return

    collection = setup_chroma_db(guide_chunks)

    selected_agent = select_guide_type()
    tts_engine = initialize_tts_engine(selected_agent['name'])
    model = 'llama3.2'
    messages = [{'role': 'system', 'content': selected_agent['system_prompt']}]
    options = {
        'temperature': selected_agent['temperature'],
        'max_tokens': selected_agent['max_tokens']
    }

    print(f"Welcome Behrend Student! You are now exploring Erie with: {selected_agent['name']}. Type '/exit' to end the session.")

    while True:
        user_input = input("You: ")

        if user_input.strip().lower() == '/exit':
            print("Guide: Safe travels! Enjoy your time in Erie! We Are!")
            break

        context = retrieve_context(collection, user_input)
        context_text = "\n\n".join(context) if context else "No relevant information found in the guide."
        user_input += f"\n\nContext: {context_text}"
        # Coordinates for Penn State
        behrend_coords = (42.1184, -80.0728)
        messages.append({'role': 'user', 'content': user_input})
        response = ollama.chat(model=model, messages=messages, stream=False, options=options)

        # Call to the tool based on the AI's response
        response_text = append_distance_to_coordinates(response.message.content, behrend_coords)
        print(f"Guide: {response_text}")
        text_to_speech(response.message.content, tts_engine)

if __name__ == "__main__":
    main()

