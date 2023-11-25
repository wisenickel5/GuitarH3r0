import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import re
import os
import requests


# s is input text
def normalize_text(s, sep_token=" \n "):
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r". ,", "", s)
    # remove all instances of multiple spaces
    s = s.replace("..", ".")
    s = s.replace(". .", ".")
    s = s.replace("\n", "")
    s = s.strip()
    return s


# Prompting ChatGPT to come up with a response to someone calling about an overflowing water heater
prompt = ("Thank you for calling Nexidia heat and plumbing, how may i help you? \n "
          "Hi, my water-heater is overflowing. \n"
          " have you turned off water to the house? \n"
          " No, how do i do that?")

# Setup parameters
openai.api_type = "azure"
openai.api_key = API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2022-12-01"

# Verify the OpenAI API can be accessed
url = f"{openai.api_base}/openai/deployments?api-version=2022-12-01"
r = requests.get(url, headers={"api-key": API_KEY})
print("Confirming that an entry for Embedding & Language models exist in API response:\n", r.text)

# Generate the next best sentence
model_for_completions = "gpt-35-turbo"
next_best_sentence = openai.Completion.create(
    engine=model_for_completions,
    prompt=prompt
)
print(next_best_sentence.choices[0].text)

# Two responses to compare to the one suggested by openai:
response_1 = "do you know where your water meter is located?"
response_2 = "in lousiana, i like to eat poboy sandwiches"

print(prompt)
print(normalize_text(next_best_sentence.choices[0].text))

# Generate Embeddings for sentences
model_for_embeddings = "text-embedding-ada-002"
embedding_0 = get_embedding(normalize_text(next_best_sentence.choices[0].text), engine=model_for_embeddings)
embedding_1 = get_embedding(normalize_text(response_1), engine=model_for_embeddings)
embedding_2 = get_embedding(normalize_text(response_2), engine=model_for_embeddings)

dist_0_0 = cosine_similarity(embedding_0, embedding_0)
dist_0_1 = cosine_similarity(embedding_0, embedding_1)
dist_0_2 = cosine_similarity(embedding_0, embedding_2)

print(f'dist(em0,em0) = {dist_0_0:0.3}')
print(f'dist(em0,em1) = {dist_0_1:0.3}')
print(f'dist(em0,em2) = {dist_0_2:0.3}')
