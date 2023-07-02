class CountEncoder(Encoder):
    '''
    Lightly edited versions from Optimized Analytics Package for Spark Platform (OAP)
    '''    
    def __init__(self, x_col_list, y_col_list, out_col_list):
        self.op_name = "CountEncoder"
        self.x_col_list = x_col_list
        self.y_col_list = y_col_list
        self.out_col_list = out_col_list
        self.expected_list_size = len(y_col_list)
        if len(self.out_col_list) < self.expected_list_size:
            raise ValueError("CountEncoder __init__, input out_col_list should be same size as y_col_list")

    def transform(self, df):
        x_col = self.x_col_list
        cols = [x_col] if isinstance(x_col, str) else x_col
        agg_all = df.groupby(cols)

        all_list = []

        for i in range(0, self.expected_list_size):
            y_col = self.y_col_list[i]
            out_col = self.out_col_list[i]
            all_list.append(F.count(y_col).alias(f'{out_col}'))

        agg_all = agg_all.agg(*all_list)
        for i in range(0, self.expected_list_size):
            out_col = self.out_col_list[i]
            agg_all = agg_all.withColumn(out_col, F.col(out_col).cast(IntegerType()))
        return agg_all


class TargetEncoder(Encoder):
'''
Lightly edited versions from Optimized Analytics Package for Spark Platform (OAP);
Presupposes this column exists:
df = df.withColumn("fold", F.round(F.rand(seed=42) * 10))
'''
def __init__(self, x_col_list, y_col_list, out_col_list, y_mean_list=None, smooth=20, seed=42,threshold=0):
    self.op_name = "TargetEncoder"
    self.x_col_list = x_col_list
    self.y_col_list = y_col_list
    self.out_col_list = out_col_list
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

        agg_per_fold = agg_per_fold.drop(f'count_all_{y_col}', f'count_{y_col}', f'sum_all_{y_col}', f'sum_{y_col}')
        agg_all = agg_all.drop(f'count_all_{y_col}', f'sum_all_{y_col}')
    return agg_all