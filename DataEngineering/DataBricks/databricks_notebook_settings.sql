set spark.databricks.delta.properties.defaults.autoOptimize.optimizeWrite = true;
set spark.databricks.delta.properties.defaults.autoOptimize.autoCompact = true;
set spark.databricks.io.cache.enabled = true;
set spark.databricks.adaptive.autoOptimizeShuffle.enabled = true; -- do not use if you want to manually set spark.sql.shuffle.partitions
set spark.databricks.delta.retryWriteConflict.enabled = true;
set spark.databricks.delta.retryWriteConflict.limit = 3;
set spark.sql.adaptive.skewJoin.enabled	= true;
set spark.sql.adaptive.enabled = true;

spark.conf.set("spark.sql.shuffle.partitions", 96);  -- num cores
spark.conf.set("spark.sql.files.maxPartitionBytes", 1024 * 1024 * 16)  -- if we dont have big data on the shuffle read; scale down file sizes from 128mb blocks to get more parallelism

-- spark config for a cluster with 432gb memory and 64 cores on a single worker with 1 driver @ 32GB / 4 cores
spark.storage.level MEMORY_AND_DISK_SER
spark.driver.extraJavaOptions -XX:+UseG1GC
spark.dynamicAllocation.enabled false
spark.serializer org.apache.spark.serializer.KryoSerializer
spark.databricks.delta.preview.enabled true
spark.rpc.message.maxSize 2047
spark.databricks.driver.disableScalaOutput true
spark.shuffle.spill.compress true
spark.driver.memoryOverhead 3174M
spark.executor.cores 5
spark.executor.memory 31G
spark.rdd.compress true
spark.executor.instances 11
spark.driver.memory 25G
spark.executor.extraJavaOptions -XX:+UseG1GC
spark.sql.adaptive.enabled true
spark.default.parallelism 110
spark.memory.fraction 0.8
spark.driver.cores 3
spark.executor.memoryOverhead 3174M