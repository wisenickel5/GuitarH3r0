from openai import AzureOpenAI
from openai.types import CreateEmbeddingResponse
import numpy as np
import re
import os
import requests
from typing import List


def normalize_text(s, sep_token=" \n ") -> str:
    """
    Removes redundant whitespace and cleans up the punctuation to prepare the data for tokenization
    :parameter: s (str): Input Text
    """
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r". ,", "", s)
    # Remove all instances of multiple spaces
    s = s.replace("..", ".")
    s = s.replace(". .", ".")
    s = s.replace("\n", "")
    s = s.strip()
    return s


def get_embedding(text: str, c: AzureOpenAI, model="Guitar-H3r0-Embeddings") -> List[float]:
    text = normalize_text(text)
    response: CreateEmbeddingResponse = c.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.
    :parameter: vec1 (array): A numpy array representing the first vector.
    :parameter: vec2 (array): A numpy array representing the second vector.

    :return: float: cosine similarity between vec1 and vec2.
    """
    # Compute the dot product between the two vectors
    dot_product = np.dot(vec1, vec2)
    # Since OpenAI embeddings are normalized, the vectors' magnitudes are both 1,
    # so we don't need to divide by the product of magnitudes.
    return dot_product


# Setup parameters
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

# Prompting ChatGPT to come up with a response to someone calling about an overflowing water heater
messages = [
    {"role": "system", "content": "Thank you for calling Nexidia heat and plumbing, how may i help you?"},
    {"role": "user", "content": "Hi, my water-heater is overflowing."},
    {"role": "system", "content": "Have you turned off water to the house?"},
    {"role": "user", "content": "No, how do i do that?"},
]
# Generate the next best sentence
next_best_sentence = client.chat.completions.create(model="Guitar-H3r0-GPT-Turbo", messages=messages)

# Two responses to compare to the one suggested by OpenAI:
response_1 = "Do you know where your water meter is located?"
response_2 = "In Louisiana, I like to eat Po-boy sandwiches"

print("Conversation: \n", [x["content"] for x in messages])
print("'Next-Best-Sentence' According to ChatGPT:\n", next_best_sentence.choices[0].message.content)
print("*Normalized* 'Next-Best-Sentence' According to ChatGPT:\n",
      normalize_text(next_best_sentence.choices[0].message.content))

# Generate Embeddings for sentences
embedding_0 = get_embedding(next_best_sentence.choices[0].message.content, client)
embedding_1 = get_embedding(response_1, client)
embedding_2 = get_embedding(response_2, client)

# Calculate distance between embeddings
dist_0_0 = cosine_similarity(embedding_0, embedding_0)
dist_0_1 = cosine_similarity(embedding_0, embedding_1)
dist_0_2 = cosine_similarity(embedding_0, embedding_2)

print(f'Distance (em0,em0) = {dist_0_0:0.3}')
print(f'Distance (em0,em1) = {dist_0_1:0.3}')
print(f'Distance (em0,em2) = {dist_0_2:0.3}')
