from tenacity import retry

@retry
def run_notebook():
    try:
        # run the notebook
        dbutils.notebook.run('/Users/afogarty@microsoft.com/DLT', timeout_seconds=999999, arguments={'notebook': '/Users/afogarty@microsoft.com/DLT'})

    except Exception as e:
        raise e

df = (spark.readStream.format("cloudFiles")
.option("cloudFiles.format", "csv")
.option("cloudFiles.schemaLocation", "dbfs:/mnt/general/streamtemp99/checkpoints/schema")
.option("cloudFiles.schemaEvolutionMode", "addNewColumns")
.option("cloudFiles.maxBytesPerTrigger", "5g")
.option("minPartitions", 989)
.load('/mnt/raw/2023110*.csv')
)

def clean_cols(df):
    df = df.withColumn('sourceFile', F.input_file_name())
    df = df.withColumn('partitionName', F.split(F.split('sourceFile', '/').getItem(4), '_').getItem(0) )
    return df

df = clean_cols(df)

df = df.writeStream \
.format("delta") \
.outputMode("append") \
.option("checkpointLocation", "dbfs:/mnt/general/streamtemp99/checkpoints") \
.option("mergeSchema", "true") \
.start('/mnt/Delta/Bronze/ML/RandomizedTest2')