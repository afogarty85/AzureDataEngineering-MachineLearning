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
    current_version: int ; whether we want to specify it
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
