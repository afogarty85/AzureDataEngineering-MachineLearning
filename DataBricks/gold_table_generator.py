def generate_gold_table(df, table_name, origin, partition):
    '''

    GOLD TABLE GENERATOR

    params:
    df: dataframe ; cleaned Gold-level dataframe ready for publishing
    table_name: string ; name of the table to be generated
    partition: string ; column we want to partition the data; default None
    origin: string ; the origin of the data (e.g., CHIE, AvailabilityPlatform, OnePDM, etc)
    '''

    # drop ETL cols
    df = df.drop('sourceFile')
    df = df.drop('createdDate')
    df = df.drop('updatedDate')

    # gen col list
    col_list = [col for col in df.columns]

    # add protections if the col names has values in it that parquet does not like
    col_list = [f"`{col}`" if ('.' in col) or ('/' in col) or (' ' in col) else col for col in col_list]

    # get dtypes
    dtype_list = [dict(df.dtypes)[col] for col in df.columns]

    # gen string
    statement_string = ''
    col_lens = len(col_list) - 1
    count = 0
    for col, type in zip(col_list, dtype_list):
        if count == col_lens:
            statement_string += f'{col} {type}'
        else:
            statement_string += f'{col} {type},' + '\n'
        count += 1


    # generate create table statement
    out = f'''
        CREATE TABLE dGold.{table_name} (
        {statement_string}
        )
         USING DELTA
         OPTIONS (PATH "/mnt/Delta/Gold/{origin}/{table_name}")
        '''

    if partition:
        out += f'PARTITIONED BY ({partition})'
    else:
        out += ''

    # run create table
    spark.sql(f'DROP TABLE IF EXISTS dGold.{table_name};')
    spark.sql(f'{out}')



# params
df = spark.read.format('parquet').load('/mnt/folder_path/*.parquet"')
table_name = 'RackLayout'
origin = 'OnePDM'
partition = None

# execute
generate_gold_table(df, table_name, origin, partition)
