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

-- spark config
spark.storage.level MEMORY_AND_DISK_SER
spark.driver.extraJavaOptions -XX:+UseG1GC
spark.dynamicAllocation.enabled true
spark.shuffle.service.enabled true
spark.serializer org.apache.spark.serializer.KryoSerializer
spark.databricks.delta.preview.enabled true
spark.databricks.driver.disableScalaOutput true
spark.shuffle.compress true
spark.shuffle.spill.compress true
spark.driver.memoryOverhead 15174M
spark.executor.cores 5
spark.executor.memory 31G
spark.driver.maxResultSize 60G
spark.dynamicAllocation.minExecutors 18
spark.rdd.compress true
spark.driver.memory 52G
spark.executor.extraJavaOptions -XX:+UseG1GC
spark.sql.adaptive.enabled true
spark.dynamicAllocation.executorIdleTimeout 60s
spark.dynamicAllocation.maxExecutors 151
spark.dynamicAllocation.initialExecutors 18
spark.default.parallelism 1000
spark.memory.fraction 0.8
spark.driver.cores 5
spark.executor.memoryOverhead 3174M
spark.databricks.rpc.maxMessageSize 2047
spark.rpc.message.maxSize 2047
