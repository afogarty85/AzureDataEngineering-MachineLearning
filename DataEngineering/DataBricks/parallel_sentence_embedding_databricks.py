
# Define the Sentence Transformer model and broadcast it  
model = SentenceTransformer("avsolatorio/GIST-Embedding-v0")  
broadcast_model = spark.sparkContext.broadcast(model)  
  
# Define the output schema for the embeddings  
output_schema = ArrayType(FloatType())  
  
# Helper UDF to convert array of JSON strings and strings into a single string  
def array_to_string(array_of_strings):  
    texts = []  
    for item in array_of_strings:  
        try:  
            # Try to load the item as JSON and convert to string  
            json_obj = json.loads(item)  
            if isinstance(json_obj, dict):  # Ensure json_obj is a dictionary  
                text = ' '.join([f'{k}: {v}' for k, v in json_obj.items()])  
                texts.append(text)  
            else:  # If json_obj is not a dictionary, treat it as a string  
                texts.append(str(json_obj))  
        except json.JSONDecodeError:  
            # If not JSON, just append the string as-is  
            if item != "Sentinel Value":  # Skip the sentinel value  
                texts.append(item)  
      
    # If after processing the list is empty (all were "Sentinel Value"), use the literal "Sentinel Value"  
    return ' '.join(texts).strip() if texts else "Sentinel Value"  
  
# Register the UDF with Spark  
array_to_string_udf = udf(array_to_string, StringType())  
  
# Apply the UDF to convert ConcatSEL column into a single string column  
df = df.withColumn("ConcatSELText", array_to_string_udf(df["ConcatSEL"]))  
  
# Define the Pandas UDF for generating embeddings  
@pandas_udf(output_schema)  
def generate_embeddings_udf(text_series: pd.Series) -> pd.Series:  
    # Access the broadcasted model within the UDF  
    local_model = broadcast_model.value  
  
    # Filter out the rows where the text is not null  
    filtered_series = text_series.dropna()  
  
    # Initialize an empty list to store embeddings  
    embeddings_list = [None] * len(text_series)  
  
    if not filtered_series.empty:
        # Compute the embeddings for non-null texts  
        embeddings = local_model.encode(filtered_series.tolist(), show_progress_bar=False)  
        # Convert embeddings to list of floats  
        embeddings_list = embeddings.tolist()  
  
    # Assign the computed embeddings to the corresponding rows in embeddings_series  
    embeddings_series = pd.Series(embeddings_list, index=text_series.index)  
  
    return embeddings_series  
  
# Apply the Pandas UDF to the DataFrame to create the embeddings column  
df = df.withColumn("SELEmbeddings", generate_embeddings_udf(df["ConcatSELText"]))