from openai import AzureOpenAI
import os
import requests

from guitar_hero_utils import normalize_text, get_embedding, cosine_similarity

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
