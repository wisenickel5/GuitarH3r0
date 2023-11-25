from openai import AzureOpenAI
from openai.types import CreateEmbeddingResponse
import numpy as np
import re
import os
import requests
from typing import List


def normalize_text(s: str, sep_token: str = " \n ") -> str:
    """
    Normalizes the input text by removing extra whitespace and correcting punctuation.

    This function is typically used to prepare text data for NLP tasks like tokenization
    or embedding generation.

    :parameter s: The text to normalize.
    :parameter sep_token: A separator token used for joining text segments, defaulted to a space followed by a newline.

    :return: The normalized text string.

    The function applies several regular expressions and string operations to clean up the text:
    - Collapses multiple whitespace characters into a single space.
    - Removes any occurrences of a space followed by a comma.
    - Replaces instances of double periods or spaced periods with a single period.
    - Strips leading and trailing whitespace characters.
    """

    # Collapse one or more whitespace characters into a single space and strip leading/trailing spaces.
    s = re.sub(r'\s+', ' ', s).strip()
    # Remove occurrences of a space followed by a comma, which is typically a punctuation error.
    s = re.sub(r"\s+,", ",", s)
    # Replace instances of double periods or spaced periods with a single period.
    s = s.replace("..", ".").replace(". .", ".")
    # Remove newline characters and any additional whitespace that may have been added.
    s = s.replace(sep_token, " ").strip()
    return s


def get_embedding(text: str, c: AzureOpenAI, model="Guitar-H3r0-Embeddings") -> List[float]:
    """
    Retrieve an embedding for the given text using Azure's OpenAI service.

    :parameter text: The input text string to be converted into an embedding.
    :parameter c: An instantiated AzureOpenAI client configured to communicate with the OpenAI service.
    :parameter model: The model identifier for the deployed embedding model to use.

    :return: A list of floating-point numbers representing the text embedding.

    The function normalizes the input text by removing redundant whitespace and
    unnecessary punctuation, then requests the OpenAI API to create an embedding
    for the processed text. It returns the embedding as a list of floats.
    """

    # Normalize the input text to remove redundant spaces and fix punctuation.
    text = normalize_text(text)

    # Call the OpenAI API to create an embedding for the normalized text.
    # 'model' should correspond to a valid deployment model identifier.
    response: CreateEmbeddingResponse = c.embeddings.create(input=[text], model=model)

    # Return the embedding from the API response.
    # The API returns a list of embeddings; we take the first one since we provided a single input.
    return response.data[0].embedding


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
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

# Two responses to compare to the one suggested by ChatGPT:
response_1 = "Do you know where your water meter is located?"
response_2 = "In Louisiana, I like to eat Po-boy sandwiches"

print("\nConversation: \n", [x["content"] for x in messages])
print("\nSample Response 1: ", f'{response_1}')
print("Sample Response 2: ", f'{response_2}\n')
print("'Next-Best-Sentence' According to ChatGPT:\n", next_best_sentence.choices[0].message.content)
print("*Normalized* 'Next-Best-Sentence' According to ChatGPT:\n",
      normalize_text(next_best_sentence.choices[0].message.content))

# Generate Embeddings for sentences
gpt_response_embedding = get_embedding(next_best_sentence.choices[0].message.content, client)
response_one_embedding = get_embedding(response_1, client)
response_two_embedding = get_embedding(response_2, client)

# Calculate distance between embeddings
dist_0_0 = cosine_similarity(gpt_response_embedding, gpt_response_embedding)
dist_0_1 = cosine_similarity(gpt_response_embedding, response_one_embedding)
dist_0_2 = cosine_similarity(gpt_response_embedding, response_two_embedding)

print(f'\nDistance (gpt_response_embedding, gpt_response_embedding) = {dist_0_0:0.3}')
print(f'Distance (gpt_response_embedding, response_one_embedding) = {dist_0_1:0.3}')
print(f'Distance (gpt_response_embedding, response_two_embedding) = {dist_0_2:0.3}')
