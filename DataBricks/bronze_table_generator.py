def generate_bronze_table(table_name, origin, parquet_location, changeDataFeed=False, removeCreatedUpdatedDate=True):
    '''

    BRONZE TABLE GENERATOR

    params:
    table_name: string ; name of the table to be generated
    origin: string ; the origin of the data (e.g., CHIE, AvailabilityPlatform, OnePDM, etc)
    parquet_location: string ; the location of the parquet files for loading (e.g., /mnt/RAW/DailyArrival/DeviceManager/....)
    changeDataFeed: bool ; whether or not to use changeDataFeed, ignoring the dailyarrival procedures and loading directly from lake
    removeCreatedUpdatedDate: bool ; if True, we autodrop any reference to CreatedDate and UpdatedDate in source file so we can inject our own ETL cols with the same name
    '''

    # if we want to remove created and updated date and use our own,
    if removeCreatedUpdatedDate:
        df = spark.read.format("parquet").option("inferSchema", "true").load(f"{parquet_location}").drop("CreatedDate").drop("UpdatedDate")
    else:
        df = spark.read.format("parquet").option("inferSchema", "true").load(f"{parquet_location}")

    # gen col list
    col_list = [col for col in df.columns]

    # add protections if the col names has values in it that parquet does not like
    col_list = [f"`{col}`" if ('.' in col) or ('/' in col) or (' ' in col) else col for col in col_list]

    # get dtypes
    dtype_list = [dict(df.dtypes)[col] for col in df.columns]

    # gen string
    statement_string = ''
    for col, type in zip(col_list, dtype_list):
        statement_string += f'{col} {type},' + '\n'

    # generate create table statement
    out = f'''
        CREATE TABLE dBronze.{table_name} (
        {statement_string}
          sourceFile string
        )
         USING DELTA
         OPTIONS (PATH "/mnt/Delta/Bronze/{origin}/{table_name}")
        '''

    if changeDataFeed:
        out += f'''
        TBLPROPERTIES (delta.enableChangeDataFeed = true)
        '''

    # run create table
    spark.sql(f'DROP TABLE IF EXISTS dBronze.{table_name};')
    spark.sql(f'{out}')


# params
table_name = 'StateTransitionCM'  # RAW/History/CHIE/StateTransitionCMNew/2022-07-23/StateTransitionCM_0_1.parquet
origin = 'CHIE'
parquet_location = "mnt/RAW/History/CHIE/StateTransitionCMNew/2022-07-23/*.parquet"
changeDataFeed = False
removeCreatedUpdatedDate = True


# execute
generate_bronze_table(table_name=table_name,
                      origin=origin,
                      parquet_location=parquet_location,
                      changeDataFeed=changeDataFeed,
                      removeCreatedUpdatedDate=removeCreatedUpdatedDate)
