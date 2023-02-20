%python
def AIO_delta_table_generator(table_name, origin, file_location, primary_key, removeCreatedUpdatedDate=True, is_JSON=False, provided_df=False, gold_only=False):
    '''

    All-in-One: BRONZE/SILVER/GOLD TABLE GENERATOR

    params:
    table_name: string ; name of the table to be generated
    origin: string ; the origin of the data (e.g., CHIE, AvailabilityPlatform, OnePDM, etc)
    file_location: string ; the location of the files for loading (e.g., /mnt/RAW/DailyArrival/DeviceManager
    primary_key: string ; the name for the table's primary key
    removeCreatedUpdatedDate: bool ; if True, we autodrop any reference to CreatedDate and UpdatedDate in source file so we can inject our own ETL cols with the same name
    is_JSON: bool ; if True, we have a JSON file
    provided_df: bool ; if True, we have provided our own df
    '''
    
    if provided_df:
        # if we have our own df already, e.g, a custom gold-tier derivative
        df = provided_df
        
    else:
        # json check
        if is_JSON:
            # use json loader
            df = spark.read.json(f"{file_location}")

        else:
            # use parquet
            df = spark.read.format("parquet").option("inferSchema", "true").load(f"{file_location}")

    # if we want to remove created and updated date and use our own,
    if removeCreatedUpdatedDate:
        df = df.drop("CreatedDate").drop("UpdatedDate")
    
    # gen col list
    col_list = [col for col in df.columns]
                
    # add protections if the col names has values in it that parquet does not like
    col_list = [f"`{col}`" if ('.' in col) or ('/' in col) or (' ' in col) else col for col in col_list]
                
    # get dtypes
    dtype_list = [dict(df.dtypes)[col] for col in df.columns]
    
    # blank holding
    tier = None
                
    # gen string
    statement_string = ''
    for col, type in zip(col_list, dtype_list):
        statement_string += f'{col} {type},' + '\n'
   
    # generate modular statements
    def min_table(tier, table_name, statement_string):
        # create statement
        out = f'''
        CREATE OR REPLACE TABLE d{tier}.{table_name} (
        {statement_string}
          sourceFile string
        '''
        return out

    # generate save loc
    def min_loc(tier, origin, table_name):
        # create statement
        out = f'''
                )
                USING DELTA
                OPTIONS (PATH "/mnt/Delta/{tier}/{origin}/{table_name}")
                '''
        return out

    # generate silver/gold conditions
    def min_silver(tier, origin, table_name, primary_key, removeCreatedUpdatedDate):
        # create statement
        if removeCreatedUpdatedDate:
            
            out = f'''
                    ,
                    {primary_key} bigint GENERATED ALWAYS AS IDENTITY (START WITH 1 INCREMENT BY 1),
                    currentVersion tinyint,
                    effectiveDate int,
                    expirationDate int
                    '''   
        else:
            out = f'''
                    ,
                    {primary_key} bigint GENERATED ALWAYS AS IDENTITY (START WITH 1 INCREMENT BY 1),
                    currentVersion tinyint,
                    effectiveDate int,
                    expirationDate int,
                    createdDate timestamp,
                    updatedDate timestamp                    
                    '''   
            
        
        return out

    # generate silver/gold conditions
    def min_gold(tier, origin, table_name, primary_key, removeCreatedUpdatedDate):
        # create statement
        if removeCreatedUpdatedDate:
            
            out = f'''
                    ,
                    {primary_key} bigint,
                    currentVersion tinyint,
                    effectiveDate int,
                    expirationDate int
                    '''   
        else:
            out = f'''
                    ,
                    {primary_key} bigint,
                    currentVersion tinyint,
                    effectiveDate int,
                    expirationDate int,
                    createdDate timestamp,
                    updatedDate timestamp                    
                    '''   
        
        return out
    
    def col_map():    
        # create statement
        out = f'''
            TBLPROPERTIES (
              'delta.columnMapping.mode' = 'name',
              'delta.enableChangeDataFeed' = 'true',
              'delta.minReaderVersion' = '2',
               'delta.minWriterVersion' = '5');
            '''
        return out

    def deepclone_table(tier, origin, table):
        '''
        Send new table structure across envs
        '''
        to_uat = f""" CREATE OR REPLACE TABLE delta.`/mnt/uat/Delta/{tier}/{origin}/{table}` DEEP CLONE d{tier}.{table}  """
        to_prd = f""" CREATE OR REPLACE TABLE delta.`/mnt/prd/Delta/{tier}/{origin}/{table}` DEEP CLONE d{tier}.{table}  """
        return to_uat, to_prd
    
    for tier in ['Bronze', 'Silver', 'Gold']:
        # create string holder
        table_statement = ''

        if tier == 'Bronze' and not gold_only:
            print(f'Now generating: d{tier}.{table_name}...')
            table_statement += min_table(tier, table_name, statement_string) + min_loc(tier, origin, table_name) + col_map()

            # send statement
            spark.sql(f'{table_statement}')
            
            # deep clone
            to_uat, to_prd = deepclone_table(tier=tier, origin=origin, table=table_name)
            spark.sql(to_uat)
            spark.sql(to_prd)

        if tier  == 'Silver' and not gold_only:
            print(f'Now generating: d{tier}.{table_name}...')
            table_statement += min_table(tier, table_name, statement_string) + min_silver(tier, origin, table_name, primary_key, removeCreatedUpdatedDate) + min_loc(tier, origin, table_name) + col_map()
 
            # send statement
            spark.sql(f'{table_statement}')
           
            # deep clone
            to_uat, to_prd = deepclone_table(tier=tier, origin=origin, table=table_name)
            spark.sql(to_uat)
            spark.sql(to_prd)

        if tier  == 'Gold':
            print(f'Now generating: d{tier}.{table_name}...')
            table_statement += min_table(tier, table_name, statement_string) + min_gold(tier, origin, table_name, primary_key, removeCreatedUpdatedDate) + min_loc(tier, origin, table_name) + col_map()
            
            # send statement
            spark.sql(f'{table_statement}')

            # deep clone
            to_uat, to_prd = deepclone_table(tier=tier, origin=origin, table=table_name)
            spark.sql(to_uat)
            spark.sql(to_prd)
