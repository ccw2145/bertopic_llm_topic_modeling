# Databricks notebook source
# MAGIC %%capture
# MAGIC %pip install bertopic openai

# COMMAND ----------

CATALOG = 'cindy_demo_catalog'
SCHEMA = 'airline_bookings'
INTENTS_TABLE = "raw_intents_1000_function"
OUTPUT_TABLE = 'labeled_reviews_output_1000_all_airlines'

TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
WORKSPACE_URL = f'https://{spark.conf.get("spark.databricks.workspaceUrl")}'

MODEL_ID = 'databricks-meta-llama-3-1-70b-instruct'


# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Reviews from UC

# COMMAND ----------

reviews_df = spark.table(f"{CATALOG}.{SCHEMA}.{INTENTS_TABLE}").toPandas()

# COMMAND ----------

all_reviews = [f"{review['intent']}: {review['text_summary']}" for _, review in reviews_df.iterrows()]

# COMMAND ----------

all_reviews, len(all_reviews)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute Embeddings

# COMMAND ----------

from sentence_transformers import SentenceTransformer

# Pre-calculate embeddings
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
embeddings = embedding_model.encode(all_reviews, show_progress_bar=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Option to save embeddings so dont have to generate embeddings everytime

# COMMAND ----------

# DBTITLE 1,Save Embeddings (optional)
# specify path to save embeddings
embedding_file_path = 'embeddings_reviews.npy'
import numpy as np
with open(embedding_file_path, 'wb') as f:
    np.save(f, embeddings)

# COMMAND ----------

# import numpy as np
# embedding_file_path = 'embeddings.npy'
# embeddings = np.load(embedding_file_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Zero-shot Topic Modeling

# COMMAND ----------

# MAGIC %md
# MAGIC ### Predefined list of topics

# COMMAND ----------

predefined_topics  = [
    "Check-in and Boarding",
    "Seating Comfort",
    "In-Flight Wi-Fi",
    "Cabin Cleanliness",
    "Food and Beverage",
    "Flight Attendants and Crew Services",
    "Baggage Handling",
    "Flight Disruptions and Delays",
    "Loyalty Program and benefits",
    "Pricing Transparency and Fees",
    "Safety Measures",
    "Ground Services Assistance",
    "Accessibility and special assistance"
]

# COMMAND ----------

len(predefined_topics)

# COMMAND ----------

# MAGIC %md
# MAGIC ### BERTopic Model

# COMMAND ----------

# MAGIC %md
# MAGIC #### Set up Databricks llm endpoint for topic generation

# COMMAND ----------

import openai
client = openai.OpenAI(base_url=f"{WORKSPACE_URL}/serving-endpoints/", api_key=TOKEN)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create the representation model
# MAGIC

# COMMAND ----------

# DBTITLE 1,custom llm prompt
prompt = """
Generate a short topic label based on reviews and key words describing the reviews. 

Make sure the response only contains the topic label. The response should be in the following format:
'''
topic: <topic label>
'''

Here are some examples of topic labels: 
'''
topic: Seating Comfort
topic: Food and Beverage Quality
topic: Baggage Handling
'''

Here are the reviews:
[DOCUMENTS]

Here are the keywords that are relevant to the reviews. Use them as reference for the topic label, but keep in mind the key words are not always the most representative. [KEYWORDS]

Now read the reviews and keywords carefully, and respond with the topic label that best categories the reviews. Remember to respond in the correct format.
"""

# COMMAND ----------

# import tiktoken
# Tokenizer
# tokenizer= tiktoken.encoding_for_model("gpt-3.5-turbo")

# COMMAND ----------

from bertopic.representation import OpenAI
from bertopic import BERTopic
openai_generator = OpenAI(
    client,
    model=MODEL_ID,
    chat=True,
    nr_docs=5,
    prompt=prompt
)

# COMMAND ----------

print(openai_generator.default_prompt_)

# COMMAND ----------

print(openai_generator.prompt)

# COMMAND ----------

# Temp helper function 
def fixed_topic_labels_(self):
    """Map topic IDs to their labels.
    A label is the topic ID, along with the first four words of the topic representation, joined using '_'.
    Zeroshot topic labels come from self.zeroshot_topic_list rather than the calculated representation.

    Returns:
        topic_labels: a dict mapping a topic ID (int) to its label (str)
    """
    topic_labels = {
        key: f"{key}_" + "_".join([word[0] for word in values[:4]])
        for key, values in self.topic_representations_.items()
    }
    if self._is_zeroshot():
        # Need to correct labels from zero-shot topics
        topic_id_to_zeroshot_label = {
            self.topic_mapper_.get_mappings()[topic_id]: self.zeroshot_topic_list[zeroshot_topic_idx]
            for topic_id, zeroshot_topic_idx in self._topic_id_to_zeroshot_topic_idx.items()
        }
        topic_labels.update(topic_id_to_zeroshot_label)
    return topic_labels
BERTopic.topic_labels_ = property(fixed_topic_labels_)


# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ##### *UPDATED: `representations` use less topic representation to speed things up*

# COMMAND ----------

# MAGIC %md
# MAGIC source code for zeroshot classification: https://github.com/MaartenGr/BERTopic/blob/master/bertopic/representation/_zeroshot.py
# MAGIC

# COMMAND ----------

# DBTITLE 1,Representation Model
## UPDATED
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, ZeroShotClassification
ai_representation = [MaximalMarginalRelevance(diversity=0.3), openai_generator]
# text_model_representation = [MaximalMarginalRelevance(diversity=0.3),ZeroShotClassification(predefined_topics, model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")] #, min_prob=0.8 


representations = {
    "AI_Generated": ai_representation,
    # "Classification_zeroshot": text_model_representation,
    "KeyBERT": KeyBERTInspired()  
}


# COMMAND ----------

# DBTITLE 1,Embedding model
# import openai
# from bertopic.backend import OpenAIBackend
# embedding_model = OpenAIBackend(client, 'databricks-bge-large-en')
# embedding_model.embed('test')

# COMMAND ----------

# DBTITLE 1,Dimension reduction
# from umap import UMAP
# umap_model = UMAP(n_neighbors=100, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

# COMMAND ----------

# DBTITLE 1,Clustering Model
## min_cluster_size=min_topic_size

# from hdbscan import HDBSCAN
# hdbscan_model = HDBSCAN(min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Put together pipeline and train
# MAGIC Source Code : https://github.com/MaartenGr/BERTopic/blob/master/bertopic/_bertopic.py#L3802
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Set `min_topic_size` to be higher (min sample needed for each cluster) for large datasets*

# COMMAND ----------

# DBTITLE 1,Topic Model
zeroshot_topic_list = predefined_topics
topic_model = BERTopic(
    embedding_model="BAAI/bge-base-en-v1.5",
    verbose=True,
    # umap_model=umap_model, hdbscan_model=hdbscan_model,
    # nr_topics=40, # reduces topic by clustering after topic generation
    min_topic_size=50, 
    zeroshot_topic_list=predefined_topics,
    zeroshot_min_similarity=.6,
    representation_model=representations 
)
topics,probs = topic_model.fit_transform(all_reviews, embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results

# COMMAND ----------

# MAGIC %md
# MAGIC - Name: predefined topics + top keywords for topic clusters 
# MAGIC - Default Representation: keywords based on c-TF-IDF (https://maartengr.github.io/BERTopic/algorithm/algorithm.html#5-topic-representation)
# MAGIC - AI_Generated: generates a label based on keywords and prompt 
# MAGIC - KeyBERT: key word extracted with Keybert() to compare

# COMMAND ----------

# DBTITLE 1,Inspect Topics
spark.createDataFrame(topic_model.get_topic_info()).display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Option to further reduce outliers with clustering

# COMMAND ----------

# Reduce outliers
new_topics = topic_model.reduce_outliers(all_reviews, topics)

# COMMAND ----------

# Update the model
topic_model.update_topics(all_reviews, topics=new_topics)

# COMMAND ----------

# Updated topics
spark.createDataFrame(topic_model.get_topic_info()).display()

# COMMAND ----------

# MAGIC %md
# MAGIC Probabilities:1) zeroshot topics:cosine similarity 2) rest: clustering 'confidence' (membership probability)

# COMMAND ----------

import pandas as pd
topic_names = [topic_model.get_topic_info(topic_id)['Name'][0] for topic_id in new_topics]
llm_topic_names = [topic_model.get_topic_info(topic_id)['AI_Generated'][0][0].strip("'''").strip('\n') for topic_id in new_topics]
results_df = pd.DataFrame(data={"topic_id": new_topics, "probability": probs, "topic_name": topic_names, "llm_topic_name": llm_topic_names, "document": all_reviews })
results_df

# COMMAND ----------

# Save results table to Unity Catalog
spark.createDataFrame(results_df).write.format("delta").saveAsTable(f"{CATALOG}.{SCHEma}.{OUTPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize topics

# COMMAND ----------

# Use llm labels for visualization
llm_topic_labels = {topic:values[0][0].strip("'''").strip('\n') for topic, values in topic_model.topic_aspects_['AI_Generated'].items()}
llm_topic_labels[-1] = "Outlier Topic"
topic_model.set_topic_labels(llm_topic_labels)
# llm_topic_labels

# COMMAND ----------

topic_model.visualize_topics(custom_labels=True)

# COMMAND ----------

TOPIC_NUM = 10
topic_model.visualize_barchart(top_n_topics=TOPIC_NUM, height=200,custom_labels=True)

# COMMAND ----------

topic_model.visualize_hierarchy(custom_labels=True)

# COMMAND ----------

# Visualize hierarchy with custom labels
topic_model.visualize_hierarchy()#custom_labels=True)

# COMMAND ----------

topic_model.visualize_heatmap()

# COMMAND ----------

topic_distr, _ = topic_model.approximate_distribution(["Cabin crew service: Felt like a nuisance and was deliberately ignored by male cabin crew."])
topic_distr

# COMMAND ----------

# Visualize the topic-document distribution for a single document
topic_model.visualize_distribution(topic_distr[0], custom_labels=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predict topics for new reviews

# COMMAND ----------

# topic_model.transform(new_reviews)

# COMMAND ----------

topic_distr, topic_token_distr = topic_model.approximate_distribution(all_reviews, calculate_tokens=True)

# Visualize the token-level distributions
df = topic_model.visualize_approximate_distribution(all_reviews[1], topic_token_distr[1])
df

# COMMAND ----------

new_document_topic, topic_probabilities = topic_model.transform(["the movie was great but was having some audio glitches along the way which put off the experience"])
# Get the topic ID assigned to the new document
topic_id = new_document_topic[0]
# Get the topic words for the assigned topic
topic_words = topic_model.get_topic(topic_id)
topic_string = ", ".join([word for word, _ in topic_words])
print(f"The new document is related to Topic {topic_id}: {topic_string}")
print(topic_probabilities)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Model

# COMMAND ----------

# embedding_model = "BAAI/bge-base-en-v1.5"
# topic_model.save("my_model_dir", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)

