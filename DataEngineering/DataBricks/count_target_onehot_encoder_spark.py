class CountEncoder():
    '''
    Lightly edited versions from Optimized Analytics Package for Spark Platform (OAP)
    '''    
    def __init__(self, x_col_list, y_col_list, out_col_list):
        self.op_name = "CountEncoder"
        self.x_col_list = [x_col_list] if not isinstance(x_col_list, list) else x_col_list
        self.y_col_list = [y_col_list] if not isinstance(y_col_list, list) else y_col_list
        self.out_col_list = [out_col_list] if not isinstance(out_col_list, list) else out_col_list
        self.expected_list_size = len(y_col_list)
        if len(self.out_col_list) < self.expected_list_size:
            raise ValueError("CountEncoder __init__, input out_col_list should be same size as y_col_list")

    def transform(self, df):
        agg_all = df.groupby(self.x_col_list)

        all_list = []

        for i in range(0, self.expected_list_size):
            y_col = self.y_col_list[i]
            out_col = '_'.join(self.out_col_list)
            all_list.append(F.count(y_col).alias(f'{out_col}'))

        agg_all = agg_all.agg(*all_list)
        for i in range(0, self.expected_list_size):
            out_col = '_'.join(self.out_col_list)
            agg_all = agg_all.withColumn(out_col, F.col(out_col).cast(IntegerType()))
        return agg_all
    
class TargetEncoder():
    '''
    Lightly edited versions from Optimized Analytics Package for Spark Platform (OAP)
    '''
    def __init__(self, x_col_list, y_col_list, out_col_list, y_mean_list=None, smooth=20, seed=42,threshold=0):
        self.op_name = "TargetEncoder"
        self.x_col_list = [x_col_list] if not isinstance(x_col_list, list) else x_col_list
        self.y_col_list = [y_col_list] if not isinstance(y_col_list, list) else y_col_list
        self.out_col_list = [out_col_list] if not isinstance(out_col_list, list) else out_col_list
        self.y_mean_list = y_mean_list
        self.seed = seed
        self.smooth = smooth
        self.expected_list_size = len(y_col_list)
        self.threshold = threshold
        if len(self.out_col_list) < self.expected_list_size:
            raise ValueError("TargetEncoder __init__, input out_col_list should be same size as y_col_list")      
        if y_mean_list != None and len(self.y_mean_list) < self.expected_list_size:
            raise ValueError("TargetEncoder __init__, input y_mean_list should be same size as y_col_list")        

    def transform(self, df):
        x_col = self.x_col_list
        cols = ['fold', x_col] if isinstance(x_col, str) else ['fold'] + x_col
        agg_per_fold = df.groupBy(cols)
        agg_all = df.groupBy(x_col)

        per_fold_list = []
        all_list = []

        for i in range(0, self.expected_list_size):
            y_col = self.y_col_list[i]
            per_fold_list.append(F.count(y_col).alias(f'count_{y_col}'))
            per_fold_list.append(F.sum(y_col).alias(f'sum_{y_col}'))
            all_list.append(F.count(y_col).alias(f'count_all_{y_col}'))
            all_list.append(F.sum(y_col).alias(f'sum_all_{y_col}'))

        agg_per_fold = agg_per_fold.agg(*per_fold_list)
        agg_all = agg_all.agg(*all_list)
        agg_per_fold = agg_per_fold.join(agg_all, x_col, 'left')

        if self.threshold > 0:
            agg_all = agg_all.where(F.col(f'count_all_{self.y_col_list[0]}')>self.threshold)

        for i in range(0, self.expected_list_size):
            y_col = self.y_col_list[i]
            out_col = self.out_col_list[i]
            y_mean = self.y_mean_list[i] if self.y_mean_list != None else None
            
            if y_mean is None:
                y_mean = np.array(df.groupBy().mean(y_col).collect())[0][0]
            mean = float(y_mean)
            smooth = self.smooth

            # prepare for agg_per_fold
            agg_per_fold = agg_per_fold.withColumn(f'count_all_{y_col}', F.col(f'count_all_{y_col}')-F.col(f'count_{y_col}'))
            agg_per_fold = agg_per_fold.withColumn(f'sum_all_{y_col}', F.col(f'sum_all_{y_col}')-F.col(f'sum_{y_col}'))
            agg_per_fold = agg_per_fold.withColumn(out_col, (F.col(f'sum_all_{y_col}') + F.lit(mean) * F.lit(smooth))/(F.col(f'count_all_{y_col}') + F.lit(smooth)))
            agg_all = agg_all.withColumn(out_col, (F.col(f'sum_all_{y_col}') + F.lit(mean) * F.lit(smooth))/(F.col(f'count_all_{y_col}') + F.lit(smooth)))

            # cast -- will go double otherwise
            agg_per_fold = agg_per_fold.withColumn(out_col, F.col(out_col).cast(FloatType()))
            agg_all = agg_all.withColumn(out_col, F.col(out_col).cast(FloatType()))

            agg_per_fold = agg_per_fold.drop(f'count_all_{y_col}', f'count_{y_col}', f'sum_all_{y_col}', f'sum_{y_col}')
            agg_all = agg_all.drop(f'count_all_{y_col}', f'sum_all_{y_col}')
        return agg_all



# some example code
encode_cols = [
                ['name', 'test_suite', 'pipeline_name'],
                'pipeline_job',
                'project',
                'testType',
                ]

# get mean of y
mean_y = train.groupBy().mean('y').collect()[0][0]

# set fold col for train
train = train.withColumn("fold", F.round(F.rand(seed=42) * 10))

for col in encode_cols:
    print(f"Now Count Encoding: {col}")

    # gen out column name
    if isinstance(col, list):
        out_col_temp = '_'.join(col) + '_CE'
    else:
        out_col_temp = col + '_CE'
    
    # fit
    CE_ENC = CountEncoder(x_col_list=col, y_col_list=['y'], out_col_list=out_col_temp)

    # transform
    tdf = CE_ENC.transform(train)

    # join
    train = train.join(tdf.hint('broadcast'), how='left', on=col).fillna(0, subset=out_col_temp)
    valid = valid.join(tdf.hint('broadcast'), how='left', on=col).fillna(0, subset=out_col_temp)
    test = test.join(tdf.hint('broadcast'), how='left', on=col).fillna(0, subset=out_col_temp)
    
    # fit target encoder
    print(f"Now Target Encoding: {col}")

    # gen out column name
    if isinstance(col, list):
        out_col_temp = '_'.join(col) + '_TE'
    else:
        out_col_temp = col + '_TE'
    
    # fit
    TE_ENC = TargetEncoder(x_col_list=col, y_col_list=['y'], out_col_list=out_col_temp, y_mean_list=[mean_y], smooth=10, seed=42, threshold=0)

    # transform
    tdf = TE_ENC.transform(train)

    # join
    train = train.join(tdf.hint('broadcast'), how='left', on=col).fillna(0, subset=out_col_temp)
    valid = valid.join(tdf.hint('broadcast'), how='left', on=col).fillna(0, subset=out_col_temp)
    test = test.join(tdf.hint('broadcast'), how='left', on=col).fillna(0, subset=out_col_temp)

    print('\n')


# OneHot
# set cols
one_hot_cols = [
                'pipeline_repo',
                'job_type',
                'project',
                'pipeline_name',
            ]

# convert strings to numeric
si = StringIndexer(inputCols=one_hot_cols,  outputCols=[x + '_SI' for x in one_hot_cols])

# convert to onehot
ohe = OneHotEncoder(inputCols=[x + '_SI' for x in one_hot_cols],  outputCols=[x + '_OH' for x in one_hot_cols])

# chain
pipeline = Pipeline(stages=[si, ohe])

# fit train
pipeline = pipeline.fit(train)

# transform
train = pipeline.transform(train)
valid = pipeline.transform(valid)
test = pipeline.transform(test)

# explode arr
for col in one_hot_cols:
    print(f"Now on col: {col}")

    # unpack arr and gen n-new columns based on schema metadata
    train =  train.withColumn(col + '_arr', vector_to_array(col + '_OH')) \
                .select('*', *[F.col(col + '_arr')[i].cast('tinyint').alias(col + f'_arr_{i}') for i in range( train.schema[col + '_OH'].metadata["ml_attr"]["num_attrs"] )])

    valid =  valid.withColumn(col + '_arr', vector_to_array(col + '_OH')) \
                .select('*', *[F.col(col + '_arr')[i].cast('tinyint').alias(col + f'_arr_{i}') for i in range( valid.schema[col + '_OH'].metadata["ml_attr"]["num_attrs"] )])

    test =  test.withColumn(col + '_arr', vector_to_array(col + '_OH')) \
                .select('*', *[F.col(col + '_arr')[i].cast('tinyint').alias(col + f'_arr_{i}') for i in range( test.schema[col + '_OH'].metadata["ml_attr"]["num_attrs"] )])