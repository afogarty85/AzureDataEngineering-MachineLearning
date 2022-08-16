import glob
import pyodbc
import re
import random
import string
from pyspark.sql import functions as F


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

# exclude IcM / BOM / MTx -- separate process for that
paths = [path for path in paths if 'IcM' not in path]
paths = [path for path in paths if 'MTx' not in path]
paths.remove('/dbfs/mnt/out/Gold/CHIE/BillOfMaterial')

# create dicts to help parameterize table loading
tables = [s.split('/')[-1] for s in paths if len(s.split('/')) == 7]


dim_fact_map = dict(zip(tables, tables))
fact_list = ['ListHere']

# for tables with more than 60m rows, generally speaking
clustered_columnstore_index_list = ['LargeTablesHere']

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

        # get lengths
        df1 = df.select([F.max(F.length(F.col(col))).alias(f"{col}") for col in df.columns]).limit(1).cache()
        df1 = df1.select([(df1[col] + 5).alias(f"{col}") for col in df1.columns])

        # build sql table lengths
        cols_ = ''
        loop_len = len(df.dtypes) -1
        for i, (name, type) in enumerate(df.dtypes):

            if type == 'string':
                len_ = df1.select(F.col(f'{name}')).first()[0]
                if len_ >= 8000: len_ = 'max'

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


for table_loc, (table, schema) in enumerate(dim_fact_map.items()):
    try:
        print(f"Now loading table: {table} with schema: {schema}")

        # load data from the gold out
        pathing_ = '/'.join(paths[table_loc].split('/')[2::])
        # load df
        df = spark.read.format("parquet").load(f"/{pathing_}")

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

        # write to SQL
        print(f"Now injecting table: {table}")
        df.write \
        .mode("append") \
        .format("com.databricks.spark.sqldw") \
        .option("url", f"jdbc:sqlserver://servername{step_}.sql.azuresynapse.net:1433;database=db_name;encrypt=true;trustServerCertificate=true;hostNameInCertificate=*.sql.azuresynapse.net;loginTimeout=30") \
        .option("useAzureMSI", "true") \
        .option("enableServicePrincipalAuth", "true") \
        .option("dbTable", f"{schema}.{table}") \
        .option("truncate", "true") \
        .option("tempDir", f"abfss://servername{step_}lakefs@servername{step_}lake.dfs.core.windows.net/temp") \
        .save()

    except Exception as e:
        print(e)

# close when done
cursor.close()
cnxn.close()
