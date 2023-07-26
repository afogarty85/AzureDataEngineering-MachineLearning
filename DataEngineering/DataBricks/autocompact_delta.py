from delta.tables import DeltaTable

# compact written files
spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")  # Use 128 MB as the target file size
delta_table = DeltaTable.forPath(spark, "dbfs:/mnt/Delta/Gold/ML/RandomizedTest")
delta_table.optimize().executeCompaction()

# vacuum out old
spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
delta_table.vacuum(0)