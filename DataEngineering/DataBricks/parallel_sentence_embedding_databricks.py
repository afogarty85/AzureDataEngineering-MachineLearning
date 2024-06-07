 
# Load the data from the table where OOS_Span is not zero  
df = spark.sql("""SELECT * FROM dGold.FleetDebugEmbeddings WHERE OOS_Span != 0""")  
  
# Concatenate SEL text columns  
cat_cols = ["UniqueEventDataDetails1", "UniqueEventDataDetails2", "UniqueEventDataDetails3", "UniqueEventType", "UniqueSensorType"]  
df = df.withColumn("ConcatSEL", concat(*cat_cols)).drop(*cat_cols)  
  
# Define the Sentence Transformer model and broadcast it  
model = SentenceTransformer("avsolatorio/GIST-Embedding-v0")  
broadcast_model = spark.sparkContext.broadcast(model)  
  
# Define the output schema for the embeddings  
output_schema = ArrayType(FloatType())  
  
# Define the Pandas UDF for generating embeddings  
@pandas_udf(output_schema)  
def generate_embeddings_udf(text_series: pd.Series) -> pd.Series:  
    # Access the broadcasted model within the UDF  
    local_model = broadcast_model.value  
  
    # Create an empty Series to store the embeddings  
    embeddings_series = pd.Series([None] * len(text_series), index=text_series.index)  
      
    # Filter out the rows where the text is not null  
    filtered_series = text_series[text_series.notna()]  
      
    if not filtered_series.empty:  
        # Compute the embeddings for non-null texts  
        embeddings = local_model.encode(filtered_series.tolist(), convert_to_tensor=True)  
        # Convert embeddings to list of lists of floats  
        embeddings_list = embeddings.cpu().numpy().tolist()  
          
        # Assign the computed embeddings to the corresponding rows in embeddings_series  
        embeddings_series[filtered_series.index] = embeddings_list  
  
    # Replace None with empty lists  
    embeddings_series = embeddings_series.apply(lambda x: x if isinstance(x, list) else [])  
      
    return embeddings_series  
  
# Apply the Pandas UDF to the DataFrame to create the embeddings column  
df = df.withColumn("embeddings", generate_embeddings_udf(col("ConcatSEL")))  
df = df.drop('ConcatSEL').drop('ResourceId')


df.cache()
df.count()