from functools import reduce  
from pyspark.sql import DataFrame
import json


# Define your file paths  
cot_paths = ['/mnt/chiemoaddevfs/SupportFiles/FLAN/FLAN/cot_fsopt_data',  
             '/mnt/chiemoaddevfs/SupportFiles/FLAN/FLAN/cot_zsopt_data']  
  
dialog_paths = ['/mnt/chiemoaddevfs/SupportFiles/FLAN/FLAN/dialog_fsopt_data',  
                '/mnt/chiemoaddevfs/SupportFiles/FLAN/FLAN/dialog_zsopt_data']  
  
flan_paths = ['/mnt/chiemoaddevfs/SupportFiles/FLAN/FLAN/flan_fsnoopt_data',  
              '/mnt/chiemoaddevfs/SupportFiles/FLAN/FLAN/flan_fsopt_data',  
              '/mnt/chiemoaddevfs/SupportFiles/FLAN/FLAN/flan_zsnoopt_data',  
              '/mnt/chiemoaddevfs/SupportFiles/FLAN/FLAN/flan_zsopt_data']  
  
niv_paths = ['/mnt/chiemoaddevfs/SupportFiles/FLAN/FLAN/niv2_fsopt_data',  
             '/mnt/chiemoaddevfs/SupportFiles/FLAN/FLAN/niv2_zsopt_data']  
  
t0_paths = ['/mnt/chiemoaddevfs/SupportFiles/FLAN/FLAN/t0_fsnoopt_data',  
            '/mnt/chiemoaddevfs/SupportFiles/FLAN/FLAN/t0_zsnoopt_data',  
            '/mnt/chiemoaddevfs/SupportFiles/FLAN/FLAN/t0_zsopt_data',
            '/mnt/chiemoaddevfs/SupportFiles/FLAN/FLAN/t0_fsopt_data',
            ]  
  
# Load the data into separate dataframes and sample  
cot_ds = [spark.read.parquet(ds).sample(False, 0.5, seed=42) for ds in cot_paths]  
dialog_ds = [spark.read.parquet(ds).sample(False, 0.5, seed=42) for ds in dialog_paths]  
flan_ds = [spark.read.parquet(ds).sample(False, 0.25, seed=42) for ds in flan_paths]  
niv_ds = [spark.read.parquet(ds).sample(False, 0.5, seed=42) for ds in niv_paths]  
t0_ds = [spark.read.parquet(ds).sample(False, 0.25, seed=42) for ds in t0_paths]


# Union all sampled dataframes within each category  
cot_union = reduce(DataFrame.unionAll, cot_ds)  
dialog_union = reduce(DataFrame.unionAll, dialog_ds)  
flan_union = reduce(DataFrame.unionAll, flan_ds)  
niv_union = reduce(DataFrame.unionAll, niv_ds)  
t0_union = reduce(DataFrame.unionAll, t0_ds)  
  
# Union all  
flan2022_submix = reduce(DataFrame.unionAll, [flan_union, t0_union, niv_union, cot_union, dialog_union]) # 144m samples

# reduce to approximately with this frac
df_size = 10000000
task_proportions = {    
    'flan': 0.4,    
    't0': 0.32,    
    'niv2': 0.20,    
    'cot': 0.05,    
    'dialog': 0.03    
}  


# Compute the size of each category  
task_sizes = {k: int(v * df_size) for k, v in task_proportions.items()}  

# Sample the dataframes according to the computed sizes  
cot_sample = cot_union.sample(True, task_sizes['cot'] / cot_union.count(), seed=42)    # cot is tiny -- sample w/ replacement
dialog_sample = dialog_union.sample(False, task_sizes['dialog'] / dialog_union.count(), seed=42)  
flan_sample = flan_union.sample(False, task_sizes['flan'] / flan_union.count(), seed=42)  
niv_sample = niv_union.sample(False, task_sizes['niv2'] / niv_union.count(), seed=42)  
t0_sample = t0_union.sample(False, task_sizes['t0'] / t0_union.count(), seed=42)  

# Union all the sampled dataframes  
df = cot_sample.union(dialog_sample).union(flan_sample).union(niv_sample).union(t0_sample).repartition(32)

# Convert the DataFrame to an RDD  
rdd = df.rdd.map(lambda x: json.dumps(x.asDict()))  
  
# Save the RDD as a text file  
rdd.coalesce(32).saveAsTextFile('/mnt/chiemoaddevfs/SupportFiles/FLANSmall/flan2022_submix')  

%sh
# Specify the directory  
DIR="/dbfs/mnt/chiemoaddevfs/SupportFiles/FLANSmall/flan2022_submix"  
  
# Loop over every file in the directory  
for FILE in "$DIR"/*  
do  
  # Rename the file to add the .jsonl extension  
  mv "$FILE" "${FILE}.jsonl"  
done  



# data available at: https://huggingface.co/datasets/BadDepartment/FLAN-Small