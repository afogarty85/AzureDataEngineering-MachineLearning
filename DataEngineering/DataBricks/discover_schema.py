from pyspark.sql.types import ByteType, ShortType, IntegerType, LongType, StringType
import pyspark.sql.functions as F


# find max values
df1 = df.select([F.max(F.col(c)).alias(c) for c in df.columns])

# create a dict of the max value
df_col_max_vals = df1.toPandas().to_dict('records')[0]
df_col_max_vals = sc.broadcast(df_col_max_vals)

# remove any <missing>
df = df.select([F.when(F.col(c) == "<missing>", F.lit(None)).otherwise(F.col(c)).alias(c) for c in df.columns])

# list comprehension to generate new column expressions
new_columns = [F.when(F.col(c).isNull(), 0).otherwise(1).cast(ByteType()).alias(c + "_attnmask") for c in df.columns if c.startswith("INPUT_")]
df = df.select("*", *new_columns)

# Define dict to hold columns and their new types  
column_types = {}  

# Lists to hold new columns and fill values
new_columns = []
fill_values = {}

for column in [c for c in df.columns if '_attnmask' not in c]:
    # Get max value of column
    max_val = df_col_max_vals.value[column]
  
    if max_val is not None:  
        try:    
            max_val = int(max_val)    
        except ValueError:    
            pass  
  
        # Check if max_val is numeric  
        if isinstance(max_val, (int, float)):  
            max_val = int(max_val)  
  
            # Determine smallest possible type for the column    
            if max_val <= 127:    
                column_types[column] = ByteType()    
            elif max_val <= 32767:    
                column_types[column] = ShortType()    
            elif max_val <= 2147483647:    
                column_types[column] = IntegerType()    
            else:    
                column_types[column] = LongType()  
              
            fill_values[column] = 0  
        else:  
            column_types[column] = StringType()  
            fill_values[column] = "<missing>"  
    else:  
        column_types[column] = StringType()  
        fill_values[column] = "<missing>"  
  
    # Fill null values and cast column type at the same time
    new_columns.append(F.when(F.col(column).isNull(), F.lit(fill_values[column])).otherwise(F.col(column)).cast(column_types[column]).alias(column))

# Add original columns back to the list
mask_columns = [c for c in df.columns if c.endswith('_attnmask')]
for column in mask_columns:
    new_columns.append(F.col(column))

# Apply all transformations at once
df = df.select(*new_columns)








distinct_counts = train.agg(*(F.countDistinct(c).alias(c) for c in train.columns)).collect()[0]  
train_count = train.count()

binary_cols = []  
categorical_cols = []  
continuous_cols = []
other_cols = []
mask_cols = [c for c in train.columns if (c.startswith('INPUT_')) and (c.endswith('_attnmask'))]

# Loop over columns  
for column in [c for c in train.columns if (c.startswith('INPUT_')) and not (c.endswith('_attnmask'))]:
    # Don't do these
    if column in ['SYNDROME']:
        continue
    n = distinct_counts[column]
  
    # If column has only two distinct values, consider it binary  
    if n == 2:
        binary_cols.append(column)  

    # If column has exactly 101 distinct values, consider it continuous
    elif n == 101:
        continuous_cols.append(column)

    # If column has more than two but less than 5% unique values, consider it categorical  
    elif n > 2 and n / train_count < 0.05:  
        categorical_cols.append(column)

    # If column doesn't fit into any of above categories, consider it other
    else:
        other_cols.append(column)

print('mask col', len(mask_cols))
print('binary_cols', len(binary_cols))
print('categorical_cols', len(categorical_cols))
print('continuous_cols', len(continuous_cols))
print('other_cols', len(other_cols))