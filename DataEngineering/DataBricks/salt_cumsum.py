def salting_cumsum(df, output_feature_name, sum_value, salt_size, order_col, partition_cols):

    # Add timestamp-based salt column to the dataframe
    df = df.withColumn("salt", F.floor(F.col('timestamp') / F.lit(salt_size)))

    # add salt
    partition_cols_salt = partition_cols + ['salt']

    # Get partial cumulative sums
    window_salted = Window.partitionBy(*partition_cols_salt).orderBy(order_col)
    df = df.withColumn("cumulative_sum", F.sum(sum_value).over(window_salted))

    # Get partial cumulative sums from previous windows
    df2 = df.groupby(*partition_cols_salt).agg(F.sum(sum_value).alias("cumulative_sum_last"))
    window_full = Window.partitionBy(*partition_cols).orderBy("salt")
    df2 = df2.withColumn("previous_sum", F.lag("cumulative_sum_last", default=0).over(window_full))
    df2 = df2.withColumn("previous_cumulative_sum", F.sum("previous_sum").over(window_full))

    # Join previous partial cumulative sums with original data
    df = df.join(df2, [*partition_cols_salt])  # maybe F.broadcast(df2) if it is small enough

    # Increase each cumulative sum value by final value of the previous window
    df = df.withColumn(output_feature_name, F.col('cumulative_sum') + F.col('previous_cumulative_sum'))

    df = df.drop('previous_sum', 'cumulative_sum', 'previous_cumulative_sum', 'cumulative_sum_last')

    return df
