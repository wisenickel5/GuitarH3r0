from openai import AzureOpenAI
import os
import requests
from pathlib import Path
import pandas as pd

from guitar_hero_utils import normalize_text, get_embedding, cosine_similarity
from transcript_utils import (get_transcript_data, get_transcript_turns, create_transcript_subsets,
                              convert_subsets_to_messages)

# Setup parameters to authenticate with OpenAI API
api_key = os.getenv("AZURE_OPENAI_API_KEY")
base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
client = AzureOpenAI(
    api_version="2023-07-01-preview",
    api_key=api_key,
    azure_endpoint=base_url
)

# Verify the OpenAI API can be accessed
url = f"{base_url}/openai/deployments?api-version=2022-12-01"
r = requests.get(url, headers={"api-key": api_key})
print("Confirming that an entry for Embedding & Language models exist in API response:\n", r.text)

# Parse transcript
transcript_path = Path('/Users/dylanalexander/Repos/GuitarH3r0/Call-Center-Transcript.CSV')
transcript_df: pd.DataFrame = get_transcript_data(str(transcript_path))
turns, speakers = get_transcript_turns(transcript_df)
transcript_subsets = create_transcript_subsets(turns, speakers)
all_transcript_messages = convert_subsets_to_messages(transcript_subsets)

# Generate the next best sentence with ChatGPT
for message, actual_agent_response in zip(all_transcript_messages, ):
    next_best_sentence = client.chat.completions.create(model="Guitar-H3r0-GPT-Turbo", messages=message)

    print("\nConversation: \n", [x["content"] for x in message])
    print("\nActual Agent Response: ", f'{actual_agent_response}')
    print("'Next-Best-Sentence' According to ChatGPT:\n", next_best_sentence.choices[0].message.content)
    print("*Normalized* 'Next-Best-Sentence' According to ChatGPT:\n",
          normalize_text(next_best_sentence.choices[0].message.content))

    # Generate Embeddings for sentences
    gpt_response_embedding = get_embedding(next_best_sentence.choices[0].message.content, client)
    actual_agent_response_embedding = get_embedding(actual_agent_response, client)

    # Calculate distance between embeddings
    dist_0_0 = cosine_similarity(gpt_response_embedding, gpt_response_embedding)
    dist_0_1 = cosine_similarity(gpt_response_embedding, actual_agent_response_embedding)

    print(f'\nDistance (gpt_response_embedding, gpt_response_embedding) = {dist_0_0:0.3}')
    print(f'Distance (gpt_response_embedding, actual_agent_response_embedding) = {dist_0_1:0.3}')
