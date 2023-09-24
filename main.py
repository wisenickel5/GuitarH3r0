from azure.identity import DefaultAzureCredential
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import re
import tiktoken


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


openai_resource = "oa-research"

## For Incontact use ic-oa-research
## For Actimize  use act-oa-research


model_for_completions = "text-davinci-003"
model_for_embeddings = "text-embedding-ada-002"
prompt = ("Thank you for calling nexidia heat and plumbing, how may i help you? \n "
          "Hi, my waterheater is overflowing. \n"
          " have you turned off water to the house? \n"
          " No, how do i do that?")

# Request credential
default_credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
token = default_credential.get_token("https://cognitiveservices.azure.com/.default",
                                     exclude_shared_token_cache_credential=True)

# Setup parameters
openai.api_type = "azure_ad"
openai.api_key = token.token
openai.api_base = "https://{0}.openai.azure.com/".format(openai_resource)
openai.api_version = "2023-05-15"
tokenizer = tiktoken.get_encoding("cl100k_base")
next_best_sentence = openai.Completion.create(
    engine=model_for_completions,
    prompt=prompt
)

print(next_best_sentence.choices[0].text)
# two responses to compare to the one suggested by openai:
response_1 = "do you know where your water meter is located?"
response_2 = "in lousiana, i like to eat poboy sandwiches"

print(prompt)
print(normalize_text(next_best_sentence.choices[0].text))

embedding_0 = get_embedding(normalize_text(next_best_sentence.choices[0].text), engine=model_for_embeddings)
embedding_1 = get_embedding(normalize_text(response_1), engine=model_for_embeddings)
embedding_2 = get_embedding(normalize_text(response_2), engine=model_for_embeddings)

dist_0_0 = cosine_similarity(embedding_0, embedding_0)
dist_0_1 = cosine_similarity(embedding_0, embedding_1)
dist_0_2 = cosine_similarity(embedding_0, embedding_2)

print(f'dist(em0,em0) = {dist_0_0:0.3}')
print(f'dist(em0,em1) = {dist_0_1:0.3}')
print(f'dist(em0,em2) = {dist_0_2:0.3}')
