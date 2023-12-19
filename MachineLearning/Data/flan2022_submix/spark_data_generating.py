from functools import reduce  
from pyspark.sql import DataFrame
import json

# Define your file paths  
cot_paths = ['/mnt/chiemoaddevfs/SupportFiles/FLAN/JSONL/cot_fs_opt_train.jsonl',  
             '/mnt/chiemoaddevfs/SupportFiles/FLAN/JSONL/cot_zs_opt_train.jsonl']  
  
dialog_paths = ['/mnt/chiemoaddevfs/SupportFiles/FLAN/JSONL/dialog_fs_opt_train.jsonl',  
                '/mnt/chiemoaddevfs/SupportFiles/FLAN/JSONL/dialog_zs_opt_train.jsonl']  
  
flan_paths = ['/mnt/chiemoaddevfs/SupportFiles/FLAN/JSONL/flan_fs_noopt_train.jsonl',  
              '/mnt/chiemoaddevfs/SupportFiles/FLAN/JSONL/flan_fs_opt_train.jsonl',  
              '/mnt/chiemoaddevfs/SupportFiles/FLAN/JSONL/flan_zs_noopt_train.jsonl',  
              '/mnt/chiemoaddevfs/SupportFiles/FLAN/JSONL/flan_zs_opt_train.jsonl']  
  
niv_paths = ['/mnt/chiemoaddevfs/SupportFiles/FLAN/JSONL/niv2_fs_opt_train.jsonl',  
             '/mnt/chiemoaddevfs/SupportFiles/FLAN/JSONL/niv2_zs_opt_train.jsonl']  
  
t0_paths = ['/mnt/chiemoaddevfs/SupportFiles/FLAN/JSONL/t0_fs_noopt_train.jsonl',  
            '/mnt/chiemoaddevfs/SupportFiles/FLAN/JSONL/t0_zs_noopt_train.jsonl',  
            '/mnt/chiemoaddevfs/SupportFiles/FLAN/JSONL/t0_zs_opt_train.jsonl']  
  
# Load the data into separate dataframes and sample  
cot_ds = [spark.read.json(ds).sample(False, 0.5, seed=42) for ds in cot_paths]  
dialog_ds = [spark.read.json(ds).sample(False, 0.5, seed=42) for ds in dialog_paths]  
flan_ds = [spark.read.json(ds).sample(False, 0.25, seed=42) for ds in flan_paths]  
niv_ds = [spark.read.json(ds).sample(False, 0.5, seed=42) for ds in niv_paths]  
t0_ds = [spark.read.json(ds).sample(False, 0.33, seed=42) for ds in t0_paths] # missing fsopt file; should be 25%
  
# Union all sampled dataframes within each category  
cot_union = reduce(DataFrame.unionAll, cot_ds)  
dialog_union = reduce(DataFrame.unionAll, dialog_ds)  
flan_union = reduce(DataFrame.unionAll, flan_ds)  
niv_union = reduce(DataFrame.unionAll, niv_ds)  
t0_union = reduce(DataFrame.unionAll, t0_ds)  
  
# Union all  
flan2022_submix = reduce(DataFrame.unionAll, [flan_union, t0_union, niv_union, cot_union, dialog_union]) # 144m samples

# reduce to approximately
df_size = 2000000
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
cot_sample = cot_union.sample(False, task_sizes['cot'] / cot_union.count(), seed=42)  
dialog_sample = dialog_union.sample(False, task_sizes['dialog'] / dialog_union.count(), seed=42)  
flan_sample = flan_union.sample(False, task_sizes['flan'] / flan_union.count(), seed=42)  
niv_sample = niv_union.sample(False, task_sizes['niv2'] / niv_union.count(), seed=42)  
t0_sample = t0_union.sample(False, task_sizes['t0'] / t0_union.count(), seed=42)  

# concat
df = reduce(DataFrame.unionAll, [cot_sample, dialog_sample, flan_sample, niv_sample, t0_sample])

# RDD
rdd = df.rdd.map(lambda x: json.dumps(x.asDict()))  
  
# save  
rdd.coalesce(12).saveAsTextFile('/mnt/chiemoaddevfs/SupportFiles/FLANSmall/flan2022_submix')


# resulting proportions for 2,001,039 samples
# flan: 0.3999592211845946
# t0: 0.3202011554997179
# niv2: 0.19981519600567504
# cot: 0.050002023948558724
# dialog: 0.030022403361453724
