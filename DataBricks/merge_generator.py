# generate complex merge statements quickly with this generator



# new
class MERGE_INTO_GENERATOR():
    '''
    MERGE INTO GENERATOR

    params:
    db: string ; the Hive database
    table: string ; the Hive table name
    naturalKey: string ; the natural key of the table
    changeset: string ; the name of the changeset sql table
    extraJoins: string ; the name of the changeset Hive table/view
    housekeepingColumns: list; a list of housekeeping columns you might have that should be excluded
    identityCol: string; the name of the identity (primary key) column, if exists
    dataSkipColumn: string; the column that we will use to skip data
    dataSkipCondition: string; what condition we need with the data skipping (e.g, timestamp >=, timestamp <=, etc); choose: [ge, le, g, l, eq]
    dataSkipScalar: string; the scalar we are skipping data by
    currentVersion: string; the name of the currentVersion column
    bronzeDB: string; name of the bronze database that we might need in order to check schema drift
    checkDrift: bool: whether we should check for schema drift
    '''    
    def __init__(self, db, table, naturalKey, changeSet, extraJoins, housekeepingColumns=None, identityCol=None, dataSkipColumn=None, dataSkipCondition=None, dataSkipScalar=None, currentVersion=None, bronzeDB=None, checkDrift=False):
        self.db = db
        self.table = table
        self.naturalKey = naturalKey
        self.changeSet = changeSet
        self.extraJoins = extraJoins
        self.housekeepingColumns = housekeepingColumns
        self.identityCol = identityCol
        self.columns = spark.read.table(f'{db}.{table}').columns
        self.dataSkipColumn = dataSkipColumn
        self.dataSkipCondition = dataSkipCondition
        self.dataSkipScalar = dataSkipScalar
        self.currentVersion = currentVersion
        self.bronzeDB = bronzeDB
        self.checkDrift = checkDrift

        
    if self.checkDrift:
        # get datatypes
        bronze_types = dict(spark.sql(f'select * from {self.bronzeDB}.{self.table}').dtypes)
        silver_types = dict(spark.sql(f'select * from {self.db}.{self.table}').dtypes)

        # find col drift
        print('checking for drift')
        col_drift = list(bronze_types.keys() - silver_types)

        # find drifted dtypes
        if len(col_drift) >= 1:
            print('Drift found...')
            print('checking drift type')
            type_drift = [bronze_types.get(col) for col in col_drift]

            # add protections for wacky ado naming
            print('checking renaming')
            col_drift = [f"`{col}`" if ('.' in col) or ('-' in col) else f"{col}" for col in col_drift ]

            # alter schema to include newest additions
            print('executing instructions')
            for col, dtype in zip(col_drift, type_drift):
                print(f'Now adding col: {col} with dtype {dtype} since schema has drifted')
                spark.sql(f'''ALTER TABLE {self.db}.{self.table} ADD COLUMNS ({col} {dtype})''')


    def data_skipping_condition(self, dataSkipCondition):
        condition_map = {
                        'ge': '>=',
                          'le': '<=',
                          'g': '>',
                          'l': '<',
                          'eq': '=',
                          }

        data_skipping_statement = f"AND c.{self.dataSkipColumn} {condition_map.get(self.dataSkipCondition)} '{self.dataSkipScalar}'"
        return data_skipping_statement       

    def extra_join_col_generator(self, col_list):
        ''' what other cols we want to join on besides the primary key'''
        s1 = ''
        for col in col_list:
            s1 += f'AND c.{col} = cs.{col} \n'
        return s1

    def ignore_col_change_generator(self, col_list, is_union):
        ''' what columns do we want to detect changes along? likely everything '''
        col_list = [c for c in col_list if c not in self.extraJoins]
        col_list = [c for c in col_list if c != self.identityCol]
        col_list = [c for c in col_list if c != self.naturalKey]
        col_list = [c for c in col_list if c not in self.housekeepingColumns]
        s1 = ''
        s2 = ''
        for i, col in enumerate(col_list):
            if i == 0:
                s1 += f'c.{col} = cs.{col} \n'
            else:
                s1 += f'AND c.{col} = cs.{col} \n'
        
        if is_union:
            for i, col in enumerate(col_list):
                if i == 0:
                    s2 += f'c.{col} = u.{col} \n'
                else:
                    s2 += f'AND c.{col} = u.{col} \n'       
            return s2
        else:
            return s1

    def insert_statement_col_generator(self, col_list):
        col_list = [c for c in col_list if c != self.identityCol]
        s1 = ''
        for i, col in enumerate(col_list):
            if i == len(col_list) - 1:
                s1 += col
            else:
                s1 += col + ', '
        return s1

    def generate_union(self):

        update_insert = f'''MERGE INTO {self.db}.{self.table} AS c
                                \t USING
                            (
                                SELECT -- RECORDS TO UPDATE
                                cs.{self.naturalKey} as MERGEKEY,
                                cs.*
                            FROM {self.changeSet} cs

                            UNION ALL

                            SELECT -- RECORDS TO INSRT
                            NULL as MERGEKEY,
                            cs.*
                            FROM {self.changeSet} cs
                            JOIN {self.db}.{self.table} c
                            ON c.{self.naturalKey} = cs.{self.naturalKey}

                            '''
        if self.currentVersion and self.extraJoins:
            update_insert += f'''{self.extra_join_col_generator(self.extraJoins)}
                                AND c.{self.currentVersion} = 1

                                '''
        return update_insert

    def generate_where_not(self):

        where_not = f'''WHERE NOT
        ( 
            {self.ignore_col_change_generator(self.columns, is_union=False)}
            )

            '''
        
        if self.dataSkipColumn:
            where_not += self.data_skipping_condition(self.dataSkipCondition)

        # add in closure
        where_not += ') u'
        return where_not


    def generate_match(self):

        match_condition_statement = f'''
                                    ON c.{self.naturalKey} = u.MERGEKEY

                                    '''

        if self.dataSkipColumn:
            match_condition_statement += self.data_skipping_condition(self.dataSkipCondition)

        return match_condition_statement

    def generate_match_condition(self):

        match_col_list = self.insert_statement_col_generator(self.columns)

        when_matched = f'''
                        WHEN MATCHED

                        '''
        if self.currentVersion:
            when_matched += f'''AND c.{self.currentVersion} = 1 \n '''
        
        when_matched += f'''
                        AND NOT
                        (
                            {self.ignore_col_change_generator(self.columns, is_union=True)}
                        )

                        THEN UPDATE SET
                        {self.currentVersion} = 0

                        WHEN NOT MATCHED THEN
                        INSERT ({match_col_list})
                        VALUES ({match_col_list})
                        
                        '''
        return when_matched

    def generate_merge(self):
        merge_statement = self.generate_union() + self.generate_where_not() + self.generate_match() + self.generate_match_condition()
        return merge_statement


# init class
mg = MERGE_INTO_GENERATOR(db='dSilver', table='ResourceTable', naturalKey='ResourceId', changeSet='changeset', extraJoins=['Name', 'Tenant'], 
                          housekeepingColumns=['sourceFile', 'expirationDate', 'effectiveDate', 'createdDate', 'updatedDate'],
                          identityCol='resourceTableKey', dataSkipColumn='PreciseTimeStamp', dataSkipCondition='ge', dataSkipScalar='2022-01-01',
                          currentVersion='currentVersion',
                          bronzeDB='dBronze', checkDrift=True)

# print sql statement
print(mg.generate_merge())




# old
def generate_merge_into(source_cols, changeset, table_name, natural_key, partition_col, partition_vals, extra_join_keys, housekeeping_cols, effective_date, current_version):
    '''
    MERGE INTO GENERATOR

    params:
    source_cols: list ; the columns, minus housekeeping to be evaluated
    changeset: string ; the name of the changeset sql table
    table_name: string ; the name of the table we are merging into
    natural_key: string ; the natural key
    partition_col: string ; the partition col, if exists
    partition_vals: list ; the values for the parititon in the changeset
    extra_join_keys: list ; extra keys to join/match on
    housekeeping_cols: list ; the housekeeping cols
    current_version: int ; the value we want specified in the merge statement
    '''

    num_cols = len(source_cols) - 1
    num_cols_with_hk = len(source_cols + housekeeping_cols) - 1
    last_col = source_cols.copy()
    last_col.remove(natural_key)
    last_col = last_col[-1]

    # build insert statement
    insert_statement = ''
    for i, tbl_col in enumerate(source_cols):
        if tbl_col == natural_key:
            # dont want it in the statement
            continue
        if (tbl_col == last_col):
            insert_statement += f'c.{tbl_col} = cs.{tbl_col}'
            break
        elif i != num_cols:
            insert_statement += f'c.{tbl_col} = cs.{tbl_col} \n \t AND '
        else:
            insert_statement += f'c.{tbl_col} = cs.{tbl_col} \n \t '

    # build match statement
    match_statement = ''
    for i, tbl_col in enumerate(source_cols):
        if tbl_col == natural_key:
            # dont want it in the statement
            continue
        elif (tbl_col == last_col):
            match_statement += f'c.{tbl_col} = u.{tbl_col}'
            break
        elif i != num_cols:
            match_statement += f'c.{tbl_col} = u.{tbl_col} \n \t AND '
        else:
            match_statement += f'c.{tbl_col} = u.{tbl_col} \n \t '

    # build insert column statement
    insert_cols_statement = ''
    for i, tbl_col in enumerate(source_cols + housekeeping_cols):
        if i % 4 == 0:
            insert_cols_statement += '\n'
        if i != num_cols_with_hk:
            insert_cols_statement += f'{tbl_col}, '
        else:
            insert_cols_statement += f'{tbl_col}'

    # build insert values statement
    insert_vals_statement = ''
    for i, tbl_col in enumerate(source_cols + housekeeping_cols):
        if i % 4 == 0:
            insert_vals_statement += '\n'
        if i != num_cols_with_hk:
            insert_vals_statement += f'u.{tbl_col}, '
        else:
            insert_vals_statement += f'u.{tbl_col}'

    if (partition_col != None) and (partition_vals != None):
        # create partition capability
        partition_vals = ["'" + str(s) + "'" for s in partition_vals]
        partition_vals = '(' + ', '.join(partition_vals) + ')'

    # for more complicated joins
    if extra_join_keys != None:
        join_statement_a = ''
        num_join_cols = len(source_cols) - 1
        for i, tbl_col in enumerate(extra_join_keys):
            if i % 4 == 0:
                join_statement_a += '\n'
            if i != num_join_cols:
                join_statement_a += f'AND c.{tbl_col} = cs.{tbl_col} '

        join_statement_b = ''
        num_join_cols = len(source_cols) - 1
        for i, tbl_col in enumerate(extra_join_keys):
            if i % 4 == 0:
                join_statement_b += '\n'
            if i != num_join_cols:
                join_statement_b += f'AND  c.{tbl_col} = u.{tbl_col} '

    # build statements
    merge_into_a = (f'''
                MERGE INTO {table_name} AS c
                USING
                (
                    SELECT -- UDPATE
                        cs.{natural_key} as MERGEKEY,
                        cs.*
                    FROM {changeset} cs
                UNION ALL
                    SELECT -- INSERT
                        NULL as MERGEKEY,
                        cs.*
                    FROM {changeset} cs
                    JOIN {table_name} c
                    ON c.{natural_key} = cs.{natural_key}
                    AND c.currentVersion = {current_version}
                    ''')

    merge_into_b = (f'''
                    WHERE NOT -- dont insert these
                        (
                        {insert_statement}
                        )
                    ) u
                    ''')

    merge_into_c = (f'''
            WHEN MATCHED
                AND c.currentVersion = {current_version}
                AND -- And if any of these attributes have changed
                 NOT (
                 {match_statement}
                 )
            THEN UPDATE SET -- update fields on old records
                currentVersion = 0,
                expirationDate = {effective_date}
            WHEN NOT MATCHED THEN
            -- we have to list out all fields due to identity col
            INSERT (
            {insert_cols_statement}
            )
            VALUES (
            {insert_vals_statement}
            )
            ''')

    if extra_join_keys != None:
        extra_join_on_a = (f'''{join_statement_a}''')
        extra_join_on_b = (f'''{join_statement_b}''')
    else:
        extra_join_on_a = ''
        extra_join_on_b = ''

    if partition_col != None:
        partition_join = (f'''
                        -- match record condition
                        ON c.{partition_col} IN {partition_vals}
                        AND c.{natural_key} = u.MERGEKEY
                        ''')
    else:
        partition_join = (f'''
                            -- match record condition
                            ON c.{natural_key} = u.MERGEKEY
                            ''')

    merge_into = (merge_into_a + extra_join_on_a +
              merge_into_b + partition_join + extra_join_on_b + merge_into_c)
    return merge_into


# params
out = generate_merge_into(source_cols=source_cols, # a list
                          changeset='changeset', # string
                          table_name=silver_table_, # string
                          natural_key=natural_key_, # string
                          partition_col=partition_, # string
                          partition_vals=partition_parts,
                          extra_join_keys=None, # a list
                          housekeeping_cols=['sourceFile', 'currentVersion',
                                               'effectiveDate', 'expirationDate',
                                               'createdDate', 'updatedDate'],
                         effective_date=effective_dating,
                         current_version=1)

# execute
merge_out = spark.sql(f'{out}')

# display
%python display(merge_out)
