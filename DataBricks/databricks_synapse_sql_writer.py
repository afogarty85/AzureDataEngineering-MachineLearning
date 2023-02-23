import glob
import pyodbc
import re
import random
import string
from pyspark.sql import functions as F
import time

# write files from Spark/Databricks to Azure Synapse SQL
# Automated workflow which does the following:
# 1. Identifies files for writing
# 2. Generates tables and views dynamically
# 3. Creates tables and views and writes new data to them
# 4. Uses DB secrets and keys to protect access
# 5. Uses a service principal to connect

# set step
step_ = dbutils.widgets.get('step')
print(step_)

if step_ == 'dev':
    client_id_ = 'dev-synapse-sqlpool-sp-id'
    client_secret_ = 'dev-synapse-sqlpool-sp-secret'

if step_ == 'uat':
    client_id_ = 'uat-synapse-sqlpool-sp-id'
    client_secret_ = 'uat-synapse-sqlpool-sp-secret'

if step_ == 'prd':
    client_id_ = 'prd-synapse-sqlpool-sp-id'
    client_secret_ = 'prd-synapse-sqlpool-sp-secret'


class ViewGenerating():
    '''
    Dynamic View Generator
    '''
    def __init__(self, cols, table):
        self.cols = cols
        self.attributes = []
        self.dates = []
        self.housekeeping = []
        self.table = table
        self.table_acronym = self.date_acronym_gen(self.table, date_flag=False)

    def date_acronym_gen(self, s, date_flag=True):
        ''' Just get the capitalized characters '''
        if date_flag == True:
            return 'd' + ''.join(re.findall('([A-Z])', s))
        else:
            return ''.join(re.findall('([A-Z])', s))

    def sort_columns(self):
        '''
        Sort columns for processing by type
        '''
        for col in self.cols:
            if 'DateKey' in col:
                self.dates.append(col)
            elif col in ['currentVersion']:
                self.housekeeping.append(col)
            else:
                self.attributes.append(col)

    def gen_attribute_statement(self):
        '''
        Non-date/housekeeping cols
        '''
        attribute_statement = ''
        max_atts = len(self.attributes) - 1
        for i, col in enumerate(self.attributes):
            if i != max_atts:
                attribute_statement += f"{self.table_acronym}.[{col}],\n\t"
            else:
                attribute_statement += f"{self.table_acronym}.[{col}]\n\t"
        return attribute_statement

    def gen_housekeeping_statement(self):
        '''
        Housekeeping cols; todo: sort effective/expriration cols here, too
        '''
        housekeeping_statement = ''
        for col in self.housekeeping:
            housekeeping_statement += f"{self.table_acronym}.[{col}],"
        return housekeeping_statement

    def gen_date_statement(self):
        '''
        build date generating statements
        '''
        date_statement = ''
        date_join_statement = ''
        for col in self.dates:
            # get date name
            date_name = self.date_acronym_gen(
                col) + ''.join(random.choices(string.ascii_uppercase, k=3))
            date_statement += f"CASE WHEN {date_name}.[dateKey] > 0 THEN {date_name}.date ELSE '<missing>' END AS {col},\n\t"
            date_join_statement += f"LEFT JOIN [dim].[Date] {date_name} ON {self.table_acronym}.[{col}] = {date_name}.dateKey \n\t"
        return date_statement, date_join_statement

    def build_view(self):
        '''
        Execute all
        '''
        # build cols
        self.sort_columns()
        # build gen statements
        attribute_statement = self.gen_attribute_statement()
        housekeeping_statement = self.gen_housekeeping_statement()
        date_statement, date_join_statement = self.gen_date_statement()
        table_acronym = self.date_acronym_gen(table, date_flag=False)
        return attribute_statement, housekeeping_statement, date_statement, date_join_statement, table_acronym



# Defining the service principal credentials for the Azure storage account; spark will inherit this to get into the db as well
spark.conf.set("fs.azure.account.auth.type", "OAuth")
spark.conf.set("fs.azure.account.oauth.provider.type",  "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set("fs.azure.account.oauth2.client.id", dbutils.secrets.get(scope=f"servername{step_}", key=client_id_))
spark.conf.set("fs.azure.account.oauth2.client.secret", dbutils.secrets.get(scope=f"servername{step_}", key=client_secret_))
spark.conf.set("fs.azure.account.oauth2.client.endpoint", "https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47/oauth2/token")



# get paths
pathing = []
allFiles = glob.glob("/dbfs/mnt/out/Gold/*/*") # get files
for file in allFiles:
    pathing.append(file)

# get data origins (e.g., OnePDM) and tables
paths = [s for s in pathing if len(s.split('/')) == 7]

# create dicts to help parameterize table loading
tables = [s.split('/')[-1] for s in paths if len(s.split('/')) == 7]

dim_fact_map = dict(zip(tables, tables))
fact_list = ['ListHere']

# for tables with more than 60m rows, generally speaking
clustered_columnstore_index_list = []

# set dim/fact
dim_fact_map.update((k, 'dim') for k, v in dim_fact_map.items() if k not in fact_list)
dim_fact_map.update((k, 'fact') for k, v in dim_fact_map.items() if k in fact_list)

# table map
tb_map = dict(zip(tables, paths))

# assert equality
assert len(paths) == len(dim_fact_map), 'Paths do not match dims, stopping'



total_tables = []
total_views = []
total_view_drops = []

for table_loc, (table, schema) in enumerate(dim_fact_map.items()):
    try:
        print(f"Now loading table: {table} with schema: {schema}")

        # load data from the gold out
        pathing_ = '/'.join(paths[table_loc].split('/')[2::])

        # load df
        df = spark.read.format("parquet").load(f"/{pathing_}")

        # drop annoying index col if exists
        df = df.drop('__index_level_0__')

        # get lengths
        df1 = df.select([F.max(F.length(F.col(col)) + 5).alias(f"{col}") for col in df.columns])

        # create a dict of the lengths
        df_col_lengths = df1.toPandas().to_dict('records')[0]

        # update dict to say max if we are over the limit
        df_col_lengths.update((k, 'max') for k, v in df_col_lengths.items() if v >= 8000)

        # get df count
        df_count = df.count()

        # clustered columnstore check; if greater than 60m and we dont violate max varchar
        if (df_count >= 60000000) and ( 'max' not in df_col_lengths.values() ):
            clustered_columnstore_index_list.append(table)

        # build sql table lengths
        cols_ = ''
        loop_len = len(df.dtypes) -1
        for i, (name, type) in enumerate(df.dtypes):

            if type == 'string':
                # get the length
                len_ = df_col_lengths.get(name)

            if type == 'double':
                type = 'float'

            if type == 'boolean':
                type = 'bit'

            if type == 'timestamp':
                type = 'DATETIME2 (7)'

            if i != loop_len:
                if (type == 'DATETIME2 (7)'):
                    cols_ += f'[{name}] {type}, \n'

                elif (type != 'string'):
                    cols_ += f'[{name}] [{type}], \n'

                else:
                    cols_ += f'[{name}] [varchar] ({len_}), \n'

            if i == loop_len:
                if (type == 'DATETIME2 (7)'):
                    cols_ += f'[{name}] {type}, \n'

                elif type != 'string':
                    cols_ += f'[{name}] [{type}] \n'

                else:
                    cols_ += f'[{name}] [varchar] ({len_}) \n'

        # build schema
        current_table = (f'''IF OBJECT_ID(N'[{schema}].[{table}]') IS NOT NULL DROP TABLE [{schema}].[{table}];
                    CREATE TABLE [{schema}].[{table}]
                    (
                        {cols_}
                    )
                    ''')

        # small tables get heap; leave off for clustered columnstore index
        if table not in clustered_columnstore_index_list:
            current_table += (f'''
                                WITH
                                (
                                DISTRIBUTION = REPLICATE,
                                HEAP
                                )
                                ''')

        # build view drop
        current_view_drop = (f'''IF OBJECT_ID(N'[{schema}].[v{table}]') IS NOT NULL DROP VIEW [{schema}].[v{table}];''')

        # view helpers
        attribute_statement, housekeeping_statement, date_statement, date_join_statement, table_acronym = ViewGenerating(cols=df.columns, table=table).build_view()

        # build view gen
        current_view = (f'''
                      CREATE VIEW [{schema}].[v{table}] AS (
                        SELECT
                        -- housekeeping cols
                        {housekeeping_statement}
                        -- date cols
                        {date_statement}
                        -- attributes
                        {attribute_statement}
                        -- from table
                        FROM [{schema}].[{table}] AS {table_acronym}
                        -- date joins
                        {date_join_statement}
                        );
                        ''')

        # accumulate
        total_tables.append([current_table])
        total_view_drops.append([current_view_drop])
        total_views.append([current_view])

    except Exception as e:
        print(e)

# client id / secret
CLIENTID = dbutils.secrets.get(scope=f"servername{step_}", key=f"{step_}-synapse-sqlpool-sp-id")
CLIENTSECRET = dbutils.secrets.get(scope=f"servername{step_}", key=f"{step_}-synapse-sqlpool-sp-secret")
SERVER=f'servername{step_}'
# connection string
s2="Driver={ODBC Driver 17 for SQL Server};Server=tcp:%s.sql.azuresynapse.net,1433;Database=db_name;UID=%s;PWD=%s;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=15;Authentication=ActiveDirectoryServicePrincipal" % (
    SERVER, CLIENTID, CLIENTSECRET)


def inject_table_to_sql(table_loc, table, schema):
    
    '''
    Inject table to SQL
    
    params:
    table_loc: int ; enumerated index to query slices of stored data derived above
    table: string ; the table name
    schema: string ; the table schema   
    '''
    
    print(f"Now loading table: {table} with schema: {schema}")

    # load data from the gold out
    pathing_ = '/'.join(paths[table_loc].split('/')[2::])

    # load df
    df = spark.read.format("parquet").load(f"/{pathing_}").drop('__index_level_0__')

    # make some custom fixes for now
    if table in ['ProgramQualification', 'RackCharacteristic', 'RackFeature', 'ResourceDesign', 'CloudNPI']:
        col_list = [col for col, type in df.dtypes if type == 'float']
        for col in col_list:
            df = df.withColumn(col, F.col(col).cast('double'))

    # check for excessively long strings; arbitrary but seems to be a good spot for synapse sql
    long_string_cols = total_excessive_cols[table_loc]

    # trim
    if len(long_string_cols) >= 1:
        print(f'Long string found for table: {schema}.{table}')
        for col in long_string_cols:
            df = df.withColumn(col, (F.when(F.length(col) > 400000, F.substring(col, 1, 400000)).otherwise(col)))

    # connect to the sql server
    cnxn = pyodbc.connect(s2, autocommit=True)  # to send alter cmd

    # open cursor
    cursor = cnxn.cursor()

    # execute order for generate tables:
    print(f'Now building table: {table}')
    cursor.execute(total_tables[table_loc][0])

    # execute order for drop views:
    print(f'Now dropping view: {table}')
    cursor.execute(total_view_drops[table_loc][0])

    # execute order for generate views:
    print(f'Now building view: {table}')
    cursor.execute(total_views[table_loc][0])

    # track the time it takes:
    timestamp1 = time.time()

    # write to MoAD
    print(f"Now injecting table: {table}")
    df.write \
    .mode("append") \
    .format("com.databricks.spark.sqldw") \
    .option("url", f"jdbc:sqlserver://chiemoad{step_}.sql.azuresynapse.net:1433;database=moad_sql;MARS_Connection=yes;encrypt=true;trustServerCertificate=true;hostNameInCertificate=*.sql.azuresynapse.net;loginTimeout=30") \
    .option("useAzureMSI", "true") \
    .option("enableServicePrincipalAuth", "true") \
    .option("dbTable", f"{schema}.{table}") \
    .option("truncate", "true") \
    .option("tempDir", f"abfss://moad{step_}lakefs@moad{step_}lake.dfs.core.windows.net/temp") \
    .option("maxErrors", 100) \
    .save()

    timestamp2 = time.time()
    time_Delta = (timestamp2 - timestamp1) / 60
    print("This took %.2f minutes" % time_Delta, '\n' )

    # close when done
    cursor.close()
    cnxn.close()
    return

@retry(wait=wait_exponential(multiplier=2, min=15, max=45), stop=stop_after_attempt(2))
def retry_injection(table_loc, table, schema):
    '''
    Separately wrapped fn so we can receive explicit exceptions in event
    '''
    try:
        inject_table_to_sql(table_loc=table_loc, table=table, schema=schema)
    except Exception as e:
        print(f"Found injection failure with table: {table}, retrying... {e}")


# parallel data loading into sql
from concurrent.futures import ThreadPoolExecutor

table_loc = range(len(dim_fact_map.items()))
table = [k for k in dim_fact_map.keys()]
schema = [v for v in dim_fact_map.values()]

# lets send three tables simultaneously
parallel_workers = 3
with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
    executor.map(retry_injection, table_loc, table, schema)