def generate_silver_table(table_name, origin, parquet_location, partition=None, caps_primary_key=False, removeCreatedUpdatedDate=True):
    '''

    SILVER TABLE GENERATOR

    params:
    table_name: string ; name of the table to be generated
    origin: string ; the origin of the data (e.g., CHIE, AvailabilityPlatform, OnePDM, etc)
    parquet_location: string ; the location of the parquet files for loading (e.g., /mnt/RAW/DailyArrival/DeviceManager/....)
    partition: string ; column we want to partition the data; default None
    caps_primary_key: bool ; whether or not the table has an acronym in it and we do not want to lower case it, (e.g., SNHierarchyExtendedKey -- not sNHierarchyExtendedKey)
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

    if caps_primary_key:
        primary_key = table_name[0] + table_name[1::] + 'Key'  # for cases where table is SNHierarchyExtended, etc
    else:
        primary_key = table_name[0].lower() + table_name[1::] + 'Key'

    # generate create table statement
    out = f'''
        CREATE TABLE dSilver.{table_name} (
        {primary_key} bigint GENERATED ALWAYS AS IDENTITY (START WITH 1 INCREMENT BY 1),
        {statement_string}
          sourceFile string,
          currentVersion tinyint,
          effectiveDate int,
          expirationDate int,
          createdDate timestamp,
          updatedDate timestamp
        )
         USING DELTA
         OPTIONS (PATH "/mnt/Delta/Silver/{origin}/{table_name}")
        '''

    if partition:
        out += f'PARTITIONED BY ({partition})'

    # run create table
    spark.sql(f'DROP TABLE IF EXISTS dSilver.{table_name};')
    spark.sql(f'{out}')



# params
table_name = 'StateTransitionCM'  # RAW/History/CHIE/StateTransitionCMNew/2022-07-23/StateTransitionCM_0_1.parquet
origin = 'CHIE'
parquet_location = "mnt/RAW/History/CHIE/StateTransitionCMNew/2022-07-23/*.parquet"
partition = 'DataCenterName'
caps_primary_key = False
removeCreatedUpdatedDate = True

# execute
generate_silver_table(table_name=table_name,
                      origin=origin,
                      parquet_location=parquet_location,
                      partition=partition,
                      caps_primary_key=caps_primary_key,
                      removeCreatedUpdatedDate=removeCreatedUpdatedDate)
