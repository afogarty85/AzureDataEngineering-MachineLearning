from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from itertools import chain

from pyspark.ml.classification import RandomForestClassifier
from itertools import chain

# assemble features
assembler = VectorAssembler(
    inputCols=[f.name for f in train.schema.fields if f.name.startswith('INPUT') and not isinstance(f.dataType, StringType)],
    outputCol="features")

# transform train
train = assembler.transform(train)

# recode y
si = StringIndexer(inputCols=['SYNDROME'],  outputCols=['SYNDROME_SI'], handleInvalid='keep').fit(train)

# transform train
train = si.transform(train)

# get weights
y_collect = train.select("SYNDROME_SI").groupBy("SYNDROME_SI").count().collect()
unique_y = [x["SYNDROME_SI"] for x in y_collect]
total_y = sum([x["count"] for x in y_collect])
unique_y_count = len(y_collect)
bin_count = [x["count"] for x in y_collect]
class_weights_spark = {i: ii for i, ii in zip(unique_y, total_y / (unique_y_count * np.array(bin_count)))}

# spark cant np
class_weights_spark = {int(k): float(v) for k, v in class_weights_spark.items()}

# Convert the dictionary to a list of tuples  
data_tuples = list(class_weights_spark.items())  

# Define the schema  
schema = StructType([  
    StructField("SYNDROME_SI", IntegerType(), True),  
    StructField("Weight", FloatType(), True)  
])  
  
# Convert the list of tuples to a DataFrame  
weight_df = spark.createDataFrame(data_tuples, schema=schema)

# join weights to train
train = train.join(weight_df, on=['SYNDROME_SI'], how='left')


# fit
rf = RandomForestClassifier(numTrees=1000, maxDepth=5, featuresCol="features", labelCol="SYNDROME_SI", weightCol='Weight', seed=42)  # 1.7 hours for 500 trees on train set
model = rf.fit(train)

# get results
attrs = sorted(
    (attr["idx"], attr["name"])
    for attr in (
        chain(*train.schema["features"].metadata["ml_attr"]["attrs"].values())
    )
) 

feature_imp = [
    (name, model.featureImportances[idx])
    for idx, name in attrs
    if model.featureImportances[idx]
]

sorted_by_second = sorted(feature_imp, key=lambda tup: tup[1], reverse=True)

# top 50
print(sorted_by_second[:50])

feature_set = [c[0] for c in sorted_by_second[:50]]