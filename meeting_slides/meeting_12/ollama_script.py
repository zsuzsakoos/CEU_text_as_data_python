import re
from tqdm import tqdm
import spacy
from openai import OpenAI
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE

# --- Configuration for Ollama ---
# Ensure you have Ollama running locally (ollama serve)
# And pull the necessary models (in the terminal):
#   ollama pull [model name]


OLLAMA_BASE_URL = "http://localhost:11434/v1"
CHAT_MODEL = "gpt-oss:120b-cloud"  # Change to your preferred chat model
EMBEDDING_MODEL = "embeddinggemma" # Change to your preferred embedding model

# Initialize OpenAI client to point to local Ollama instance
client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama" # Required but ignored by Ollama
)

print(f"Using Ollama at {OLLAMA_BASE_URL}")
print(f"Chat Model: {CHAT_MODEL}")
print(f"Embedding Model: {EMBEDDING_MODEL}")



def get_chat_completion(messages, model=CHAT_MODEL, max_tokens=300):
    """Simple wrapper for chat completions."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in chat completion: {e}")
        return None

def get_embedding(text, model=EMBEDDING_MODEL):
    """Generates an embedding for a text string."""
    try:
        text = text.replace("\n", " ")
        return client.embeddings.create(input=[text], model=model).data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

def extract_party(text):
    """Extracts 'Democrat' or 'Republican' from text using regex."""
    if not text: return "None"
    match = re.findall(r"(Democrat|Republican)", text, re.IGNORECASE)
    return match[0] if match else "None"


########################################################
# Simple sentiment classification

system_prompt = (
    "You are a sentiment classifier. Classify text as either 'positive' or 'negative'. "
    "Output ONLY the label."
)
user_query = "Oh wow. I really hate ice-cream!"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_query}
]

response_text = get_chat_completion(messages)
print(f"Query: {user_query}\nResponse: {response_text}\n")






########################################################
# Synthetic data generation
# https://www.cambridge.org/core/journals/political-analysis/article/abs/out-of-one-many-using-language-models-to-simulate-human-samples/035D7C8A55B237942FB6DBAD7CAA4E49



# Scenarios
age = [18, 25, 35, 45, 55, 65, 75]
gender = ["male", "female"]
education = ["some high school", "high school", "university", "graduate school"]

combinations = [(a, g, e) for a in age for g in gender for e in education]

print(f"Generating synthetic responses for {len(combinations)} profiles...")

synthetic_data = []

for age, gender, education in combinations:
    system_prompt = (
        f"You are a fictional {age}-year-old {gender} living in the US with a {education} education. "
        "You must choose a political party preference: 'Democrat' or 'Republican'. "
        "Answer with ONLY the party preference."
    )
    user_q = "Which political party do you prefer?"
    
    resp = get_chat_completion([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_q}])
    
    party = extract_party(resp)
    print(f"Profile ({age}, {gender}, {education}) -> {party}")
    synthetic_data.append({"age": age, "gender": gender, "edu": education, "party": party})




########################################################
# Embeddings & Classification

df = pd.DataFrame({
    "value": [
        "I love this movie, it's so heartwarming!", 
        "Terrible film, very scary and gory.",
        "A wonderful family experience.",
        "Nightmare fuel, I couldn't sleep.",
        "Great for kids and parents alike.",
        "Blood everywhere, pure horror."
    ],
    "polarity": ["family", "horror", "family", "horror", "family", "horror"]
})

print(f"Generating embeddings using {EMBEDDING_MODEL}...")
df["embedding"] = df["value"].apply(lambda x: get_embedding(x))

# Filter out failed embeddings
df = df[df["embedding"].map(len) > 0]


# Zero-Shot Classification
print("Performing Zero-Shot Classification...")
labels = ["family", "horror"]
label_embeddings = [get_embedding(l) for l in labels]

if all(len(e) > 0 for e in label_embeddings):
    def get_score(vec, label_vecs):
        # Cosine Similarity
        sim_family = np.dot(vec, label_vecs[0]) / (np.linalg.norm(vec) * np.linalg.norm(label_vecs[0]))
        sim_horror = np.dot(vec, label_vecs[1]) / (np.linalg.norm(vec) * np.linalg.norm(label_vecs[1]))
        return "family" if sim_family > sim_horror else "horror"

    df["prediction"] = df["embedding"].apply(lambda x: get_score(x, label_embeddings))
    print("Sample predictions:", df[["value", "polarity", "prediction"]].head())

    # Visualization (t-SNE)
    if len(df) > 3:
        print("\nVisualizing with t-SNE...")
        tsne = TSNE(n_components=2, perplexity=min(5, len(df)-1), random_state=42)
        embeddings_2d = tsne.fit_transform(np.array(df["embedding"].tolist()))
        
        colors = ["blue" if p == "family" else "red" for p in df["polarity"]]
        
        plt.figure(figsize=(8, 6))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.7)
        plt.title(f"t-SNE of Embeddings ({EMBEDDING_MODEL})")
        plt.show()
        print("Plot created (plt.show() commented out).")



# Empirical data
reviews_df = pd.read_csv("reviews.csv")

reviews_df = reviews_df.sample(600)


# create a list of reviews
reviews = reviews_df["value"].tolist()

# create a list of embeddings (THIS IS RATHER SLOW)
embeddings = [get_embedding(review) for review in reviews]

# add embeddings to the dataframe
reviews_df["embedding"] = embeddings

# write to csv
# df.to_csv("reviews_with_embeddings.csv", index=False)

# read the embeddings
# df = pd.read_csv("reviews_with_embeddings.csv")

# embeddings = df["embedding"]


# ONE-SHOT CLASSIFICATION

import numpy as np
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def label_score(embedding, label_embeddings):
   return cosine_similarity(embedding, label_embeddings[1]) - cosine_similarity(embedding, label_embeddings[0])

# one prediction
prediction = 'family' if label_score(embeddings[0], label_embeddings) > 0 else 'horror'

# all predictions
predictions = ["family" if label_score(embeddings[i], label_embeddings) > 0 else "horror" for i in range(len(embeddings))]



# Supervised classification

X = np.array(embeddings)
y = reviews_df["polarity"].tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# run logistic regression
my_model = LogisticRegression(random_state=0).fit(X_train, y_train)
my_model

# predict on the test set
y_pred = my_model.predict(X_test)

# calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# calculate f1 score
from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average='macro')


# Visualization of embeddings

tsne = TSNE(n_components=2,
            perplexity=30.0,
            random_state=42)

embeddings_array = np.array(embeddings)
embeddings_2d = tsne.fit_transform(embeddings_array)

colors = ["red", "blue"]
color_labels = ["red" if reviews_df["polarity"].tolist()[i] == "negative" else "blue" for i in range(len(embeddings))]

colormap = matplotlib.colors.ListedColormap(colors)


plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0],
                      embeddings_2d[:, 1],
                      c=color_labels,
                      alpha=0.5,
                      cmap=colormap)
plt.title("t-SNE Visualization of Embeddings")
plt.show()



# -------------------------------
# Load SOTU dataset for meaning drift analysis
# -------------------------------

df = pd.read_csv("sotu.csv")



# -------------------------------
# Extract all sentences containing the target word
# -------------------------------

# MEANING DRIFT

TARGET = "freedom"
pattern = re.compile(rf"\b{TARGET}\b", re.IGNORECASE)

rows = []

for doc, (idx, row) in zip(
        nlp.pipe(df["text"], batch_size=32, n_process=8),
        df.iterrows()
    ):
    year = row["year"]
    party = row["party"]

    for sent in doc.sents:
        sent_text = sent.text
        if pattern.search(sent_text):
            rows.append({
                "year": year,
                "party": party,
                "sentence": sent_text
            })

sents_df = pd.DataFrame(rows)
print(f"Found {len(sents_df)} sentences with the word '{TARGET}'.")


sents_df["embedding"] = [
    get_embedding(sent) for sent in tqdm(sents_df["sentence"])
]

# Drop failed embeddings if necessary
# sents_df = sents_df[sents_df["embedding"].notnull()].reset_index(drop=True)

party_year_embeddings = (
    sents_df
    .groupby(["party", "year"])["embedding"]
    .apply(lambda vecs: np.mean(np.vstack(vecs), axis=0))
    .reset_index()
)


baselines = {
    party: group["embedding"].iloc[0]
    for party, group in party_year_embeddings.groupby("party")
}

party_year_embeddings["drift"] = party_year_embeddings.apply(
    lambda row: cosine_similarity(baselines[row["party"]], row["embedding"]),
    axis=1
)


modern = party_year_embeddings[
    party_year_embeddings["party"].isin(["Democratic", "Republican"])
].copy()



drift_rows = []

for party, g in modern.groupby("party"):
    g = g.sort_values("year")
    baseline = g["embedding"].iloc[0]
    for _, row in g.iterrows():
        drift_rows.append({
            "party": party,
            "year": row["year"],
            "drift": cosine_similarity(baseline, row["embedding"])
        })

drift_df = pd.DataFrame(drift_rows)


plt.figure(figsize=(10, 5))

for party, g in drift_df.groupby("party"):
    g = g.sort_values("year")
    plt.plot(g["year"], g["drift"], marker="o", label=party)

plt.title(f"Semantic Drift of '{TARGET}' Within Parties")
plt.xlabel("Year")
plt.ylabel("Cosine similarity to party baseline")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()





def expand_embeddings(df):
    # df must contain: "party", and "embedding"
    emb_matrix = np.vstack(df["embedding"])  # shape = (n, d)
    emb_df = pd.DataFrame(emb_matrix, index=df.index)

    # Drop non-numeric columns (party)
    out = pd.concat([df.drop(columns=["embedding", "party"], errors="ignore"), emb_df], axis=1)
    return out

years = sorted(df["year"].unique())

dem_expanded = expand_embeddings(
    party_year_embeddings[party_year_embeddings.party == "Democratic"].set_index("year")
)

rep_expanded = expand_embeddings(
    party_year_embeddings[party_year_embeddings.party == "Republican"].set_index("year")
)

# Reindex to full timeline
dem_re = dem_expanded.reindex(years)
rep_re = rep_expanded.reindex(years)

# Interpolate numerically
dem_re = dem_re.infer_objects(copy=False).interpolate()
rep_re = rep_re.infer_objects(copy=False).interpolate()

def row_to_vec(row):
    return row.values.astype(float)

dem_vectors = {year: row_to_vec(dem_re.loc[year]) for year in years}
rep_vectors = {year: row_to_vec(rep_re.loc[year]) for year in years}

divergence = []

for y in years:
    D = dem_vectors[y]
    R = rep_vectors[y]
    div = 1 - cosine_similarity(D, R)
    divergence.append({"year": y, "divergence": div})

divergence_df = pd.DataFrame(divergence)



plt.figure(figsize=(12,5))
plt.plot(divergence_df["year"], divergence_df["divergence"], marker="o")
plt.title(f"Semantic Divergence Between Parties for '{TARGET}'")
plt.xlabel("Year")
plt.ylabel("Divergence (1 - cosine similarity)")
plt.grid(True)
plt.show()
