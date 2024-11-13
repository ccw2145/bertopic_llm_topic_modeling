# Databricks notebook source
# MAGIC %%capture
# MAGIC %pip install openai tenacity tqdm
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

CATALOG = "cindy_demo_catalog"
SCHEMA = "airline_bookings"
REVIEWS_TABLE = "airline_scrapped_review"
INTENTS_TABLE = "raw_intents_1000_function"

# COMMAND ----------

DATABRICKS_TOKEN = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .apiToken()
    .getOrElse(None)
)
DATABRICKS_BASE_URL = (
    f'https://{spark.conf.get("spark.databricks.workspaceUrl")}/serving-endpoints'
)
MODEL_ENDPOINT_ID="databricks-meta-llama-3-1-70b-instruct"


# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data
# MAGIC Using 1000 random reviews for demo purposes

# COMMAND ----------

sdf = spark.table(f"{CATALOG}.{SCHEMA}.{REVIEWS_TABLE}")#.limit(1000)
pdf = sdf.toPandas()

# COMMAND ----------

sample_pdf = pdf.sample(n=1000).reset_index()
all_reviews = sample_pdf['Review'].tolist()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define LLM Prompt and Tool

# COMMAND ----------

# DBTITLE 1,Tool Definition
tools = [
  {
    "type": "function",
    "function": {
        "name": "extracts_intents",
        "parameters": {
          "type": "object",
          "properties": {
            "intents": {
              "type": "array",
              "description": "List of intents identified from the customer review",
              "items": {
                "type": "object",
                "properties": {
                  "intent": {
                    "type": "string",
                    "description": "Description of the identified intent"
                  },
                  "text_summary": {
                    "type": "string",
                    "description": "Summary of the intent"
                  },
                  "sentiment": {
                    "type": "string",
                    "enum": ["Positive", "Negative", "Neutral"],
                    "description": "Sentiment of the intent"
                  },
                  "named_entities": {
                    "type": "array",
                    "items": {
                      "type": "string",
                      "description": "Named entities in the text, if any, like 'Chicago' or 'XYZ Airlines'"
                    }
                  }
                },
                "required": ["intent", "text_summary", "sentiment"]
              }
            }
          }
        }
    }
  }
]


# COMMAND ----------

# DBTITLE 1,Example Output
example_output = {
  "intents": [
    {
      "intent": "Check-in experience",
      "text_summary": "The check-in process was smooth.",
      "sentiment": "Positive",
      "named_entities": ["XYZ Airlines"]
    },
    {
      "intent": "Seating comfort",
      "text_summary": "The seating was cramped.",
      "sentiment": "Negative",
      "named_entities": []
    },
    {
      "intent": "Food quality",
      "text_summary": "The food quality was below average.",
      "sentiment": "Negative",
      "named_entities": []
    },
    {
      "intent": "Flight attendant service",
      "text_summary": "The flight attendants were very polite and helpful.",
      "sentiment": "Positive",
      "named_entities": []
    },
    {
      "intent": "Baggage issue",
      "text_summary": "I had an issue with my baggage, but it was quickly resolved.",
      "sentiment": "Neutral",
      "named_entities": []
    }
  ]
}
example_output

# COMMAND ----------

# DBTITLE 1,Prompt
prompt_template = f"""
Follow instructions below and extract intents from a customer review as a json string. DO NOT include any notes or additional information in the output.

### Instructions:
- **Identify each distinct intent** in the review as "intent". The review may contain multiple distinct intents related to different aspects of the customer's experience (e.g., service, seating, food, check-in, baggage handling). 
- **Summarize the text** associated with each intent as "text_summary".
- **Classify the sentiment** (Positive, Negative, or Neutral) of each intent as "sentiment".
- **If applicable, extract any "named entities"**, such as the airline name or specific service mentioned.
- **Return a list of intents a JSON string.** Follow the output format and use example ouput below as a reference. Make sure the JSON string is COMPLETE. Do not include additional information.

### Output Format
{{format_instructions}}

### Example Review:
"I flew with XYZ Airlines for a 6-hour flight. The check-in process was smooth, but the seating was cramped, and the food quality was below average. The flight attendants were very polite and helpful. I had an issue with my baggage, but it was quickly resolved."

### Example Output (JSON format):
{{example_output}}

### Review to analyze:
{{review}}

"""
format_instructions = {
  "intents": [
    {
      "intent": "<Intent description>",
      "text_summary": "<Summarized or specific text related to the intent>",
      "sentiment": "<Positive | Negative | Neutral>",
      "named_entities": ["<Named entities, if any>"]
    },
    {
      "intent": "<Intent description>",
      "text_summary": "<Summarized or specific text related to the intent>",
      "sentiment": "<Positive | Negative | Neutral>",
      "named_entities": ["<Named entities, if any>"]
    }
  ]
}

example_output = {
  "intents": [
    {
      "intent": "Check-in experience",
      "text_summary": "The check-in process was smooth.",
      "sentiment": "Positive",
      "named_entities": ["XYZ Airlines"]
    },
    {
      "intent": "Seating comfort",
      "text_summary": "The seating was cramped.",
      "sentiment": "Negative",
      "named_entities": []
    },
    {
      "intent": "Food quality",
      "text_summary": "The food quality was below average.",
      "sentiment": "Negative",
      "named_entities": []
    },
    {
      "intent": "Flight attendant service",
      "text_summary": "The flight attendants were very polite and helpful.",
      "sentiment": "Positive",
      "named_entities": []
    },
    {
      "intent": "Baggage issue",
      "text_summary": "I had an issue with my baggage, but it was quickly resolved.",
      "sentiment": "Neutral",
      "named_entities": []
    }
  ]
}
example_output


# COMMAND ----------

from langchain.prompts import PromptTemplate
prompt = PromptTemplate(
    input_variables=["review"],
    template=prompt_template,
    partial_variables={
        "format_instructions": format_instructions,,
         "example_output":example_output}
)


# get_request(prompt=prompt,review="I flew with XYZ Airlines for a 6-hour flight.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Inference with Tool Calling
# MAGIC Documentation Example Code: https://docs.databricks.com/en/machine-learning/model-serving/function-calling.html#notebook-example

# COMMAND ----------

# DBTITLE 1,Batch Inference Functions
 
import os
import json
import concurrent.futures
from openai import OpenAI, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception,
)  # for exponential backoff
from tqdm.notebook import tqdm
from typing import List, Optional

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url=DATABRICKS_BASE_URL
)

@retry(
    wait=wait_random_exponential(min=1, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception(RateLimitError),
)
def call_chat_model(
    prompt: str,review: str, temperature: float = 0.1, max_tokens: int = 500, **kwargs
):
    """Calls the chat model and returns the response text or tool calls."""
    chat_args = {
      "model": MODEL_ENDPOINT_ID,
    "messages": [
      {
        "role": "system",
        "content": 'You are a helpful analyst for a major airline. You help anayze customer reviews and extrarct insights.'
      },
      {
        "role": "user",
        "content": prompt.format(review=review)
      }
    ],
    "max_tokens": max_tokens,
    "temperature": temperature
  }

    chat_args.update(kwargs)

    chat_completion = client.chat.completions.create(**chat_args)

    response = chat_completion.choices[0].message
    if response.tool_calls:
        call_args = [c.function.arguments for c in response.tool_calls]
        if len(call_args) == 1:
            return call_args[0]
        return call_args
    return response.content
  
def call_in_parallel(func, prompts: List[str]) -> List:
    """Calls func(p) for all prompts in parallel and returns responses."""
    # This uses a relatively small thread pool to avoid triggering default workspace rate limits.
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = []
        for r in tqdm(executor.map(func, prompts), total=len(prompts)):
            results.append(r)
        return results


# COMMAND ----------

import pandas as pd
def extract_batch(inp: str):
    return call_chat_model(prompt=prompt,review=inp, tools=tools)
  
def results_to_dataframe(reviews: List[str], responses: List[str]):
    """Combines reviews and model responses into a dataframe for tabular display."""
    return pd.DataFrame({"Review": reviews, "Model response": responses})
  
results = call_in_parallel(extract_batch, all_reviews)
results_to_dataframe(all_reviews, results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parse llm output, dropping invalid entries
# MAGIC - Review too long, incomplete json string responses (increase max_token, or chunk the review beforehand, or specify more in prompt)
# MAGIC - Review is empty or not valid (additional preprocessing could help)

# COMMAND ----------

parsed_data = []
for index, item in enumerate(results):
    try:
        parsed_data.append({"id": index, **json.loads(item)})  # Add an id column
    except json.JSONDecodeError:
        parsed_data.append({"id": index, "intents": []})  # Add an empty intents list for invalid JSON

# Create a pandas DataFrame from the parsed data
# Explode the 'intents' column to create one row per intent
exploded_df = pd.DataFrame(parsed_data).explode('intents')

# Filter out rows where 'intents' is not a dictionary (i.e., valid JSON object)
exploded_df = exploded_df[exploded_df['intents'].apply(lambda x: isinstance(x, dict))]

# Normalize the valid 'intents' column into separate columns
results_df = pd.json_normalize(exploded_df['intents'])
results_df['id'] = exploded_df['id'].values
results_df['llm_response'] = exploded_df['intents'].values


# COMMAND ----------


merged_df = pd.merge(sample_pdf, results_df, left_on= sample_pdf.index, right_on='id', how='right')

# COMMAND ----------

merged_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Results

# COMMAND ----------

# DBTITLE 1,Write table to UC
output_df = merged_df[['id','Airline Name', 'Overall_Rating', 'Review_Title',
       'Review',  'llm_response', 'intent','text_summary',
       'sentiment', 'named_entities']].rename({'Airline Name':'Airline_Name'})

from pyspark.sql.types import ArrayType, IntegerType, DoubleType, StringType, StructField, StructType

# Define the schema for the Spark DataFrame
schema = StructType([
    StructField('id', IntegerType()),
    StructField('Airline_Name', StringType()),
    StructField('Overall_Rating', StringType()),
    StructField('Review_title', StringType()),
    StructField('Review', StringType()),
    StructField('llm_response', StringType()),
    StructField('intent', StringType()),
    StructField('text_summary', StringType()),
    StructField('sentiment', StringType()),
    StructField('named_entities', ArrayType(StringType()))
])

# Convert Pandas DataFrame to Spark DataFrame with the specified schema
output_sdf = spark.createDataFrame(output_df.dropna(), schema)


# COMMAND ----------

output_sdf.write.mode('overwrite').format("delta").saveAsTable(f"{CATALOG}.{SCHEMA}.{INTENTS_TABLE}")

# COMMAND ----------


