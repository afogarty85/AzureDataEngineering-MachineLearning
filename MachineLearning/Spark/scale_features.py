from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, MinMaxScaler
from petastorm.spark.spark_dataset_converter import _convert_vector
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import VectorUDT

# feature vectors
assembler = VectorAssembler(inputCols=feature_set, outputCol="features")
train = assembler.transform(train)
valid = assembler.transform(valid)
test = assembler.transform(test)

# scale
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True).fit(train)
train = scaler.transform(train).drop('features')
train = _convert_vector(train, 'float32')

valid = scaler.transform(valid).drop('features')
valid = _convert_vector(valid, 'float32')

test = scaler.transform(test).drop('features')
test = _convert_vector(test, 'float32')

# filter / explode
train = train.select(['SYNDROME', 'test_split', 'INPUT__YML__testname'] + [train.scaledFeatures[i].alias(feature_set[i]) for i in range(len(feature_set))])
valid = valid.select(['SYNDROME', 'test_split', 'INPUT__YML__testname'] + [valid.scaledFeatures[i].alias(feature_set[i]) for i in range(len(feature_set))])
test = test.select(['SYNDROME', 'test_split', 'INPUT__YML__testname'] + [test.scaledFeatures[i].alias(feature_set[i]) for i in range(len(feature_set))])

# write to delta
train.unionByName(valid, allowMissingColumns=True) \
    .unionByName(test, allowMissingColumns=True) \
    .write.option("mergeSchema", "true") \
    .format("delta").mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{curr_out}")

# vacuum
from delta.tables import DeltaTable

# compact written files
spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")  # Use 128 MB as the target file size
delta_table = DeltaTable.forPath(spark, "dbfs:/mnt/Delta/Gold/ML/RandomizedTest")
delta_table.optimize().executeCompaction()

# vacuum out old
spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
delta_table.vacuum(0)