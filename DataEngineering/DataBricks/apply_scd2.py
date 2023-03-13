# for Spark; situations where time series data is already obtained but
# needs to be set such that one record is current for any given day
# provides output as recommended by STAR Schema

# params
# : effec_date -- the date the row was changed, e.g., changed date/updated date
# : natural_key -- an identifer for a unit of analysis

# specify changed date
effec_date = '`System.ChangedDate`'
natural_key = 'ID'

# clone changed date
df = df.withColumn('effectiveDate', F.col(effec_date))

# current version window grouping
window_var = Window().partitionBy(natural_key)
df = df.withColumn('maxModified', F.max(effec_date).over(window_var))

# set currentVersion 1 when max date, otherwise 0
df = df.withColumn(
    "currentVersion",
    when(
        F.col("maxModified") == F.col(effec_date),
        1
    ).otherwise(0)
)

# window grouping to set expiration date and effective date
w = Window().partitionBy(natural_key).orderBy(F.col(effec_date))
df = df.select("*", F.lag("effectiveDate", 1).over(w).alias("expirationDate"))
df = df.withColumn('expirationDate', F.date_sub(df['effectiveDate'], 1))
df = df.withColumn('expirationDate', F.lag("expirationDate", -1).over(w))
df = df.withColumn('effectiveDate', F.date_format(F.col('effectiveDate'), 'yyyyMMdd').cast('integer'))
df = df.withColumn('expirationDate', F.date_format(F.col('expirationDate'), 'yyyyMMdd').cast('integer'))

# set expiration date to 20991231 for the last entry, otherwise leave the date
df = df.withColumn(
    "expirationDate",
    when(
        F.col("maxModified") == F.col(effec_date),
        20991231
    ).otherwise(F.col('expirationDate'))
)

# drop max col
df = df.drop('maxModified')
