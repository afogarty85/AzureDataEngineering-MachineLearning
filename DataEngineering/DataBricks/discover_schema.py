# transform / clean columns with unknown schema to reduce file size

# find max values
df1 = df.select([F.max(F.col(c)).alias(c) for c in df.columns])

# create a dict of the max value
df_col_max_vals = df1.toPandas().to_dict('records')[0]

# Create a list comprehension that creates a new version of each column
df = df.select([F.when(F.col(c) == "<missing>", F.lit(None)).otherwise(F.col(c)).alias(c) for c in df.columns])

# Define dict to hold columns and their new types  
column_types = {}  

# Lists to hold new columns and fill values
new_columns = []
fill_values = {}

for column in df.columns:

    # Get max value of column
    max_val = df_col_max_vals[column]

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


# Fill null values
df = df.fillna(fill_values)

# Create new columns
for column in df.columns:
    new_columns.append(F.col(column).cast(column_types[column]).alias(column))
    new_columns.append(F.when(F.col(column).isNull(), 1).otherwise(0).cast(ByteType()).alias(column + "_mask"))

# Apply all transformations at once
df = df.select(*new_columns)
