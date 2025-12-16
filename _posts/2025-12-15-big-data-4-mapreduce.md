---
layout: post
title: Big data processing concepts & MapReduce
date: 2025-12-15 18:39:00-0400
featured: false
description: Realtime and Big Data Analytics Course at NYU Courant - Conceptual Notes 4
tags: 
categories: RBDA-NYU
giscus_comments: true
related_posts: false

---

#### **Big data processing concepts**

- Parallel data processing
    - Parallel data processing reduces the execution time by dividing a single large job into multiple smaller tasks that run concurrently. Ex: a single machine with multiple processors or cores.
- Distributed data processing
    - Distributed data processing is achieved through physically separate machines that are networked together as a cluster.
- Hadoop
    - Hadoop is an open-source framework for large-scale data storage and data processing that is compatible with commodity hardware.
    - It can be used as an analytics engine for processing large amounts of structured, semi-structured and unstructured data.
    - It implements the MapReduce processing framework.
- Processing workloads
    - A processing workload in Big Data is defined as the amount and nature of data that is processed within a certain amount of time.
    - Workloads are usually divided into two types: Batch processing and transactional processing.
    - Batch processing:
        - (A.k.a. offline processing) involves processing data in batches and usually imposes delays, which results in high-latency responses.
        - Batch workloads typically involve large quantities of data with sequential read/writes and comprise of groups of read or write queries.
        - Queries can be complex and involve multiple joins.
        - OLAP (online analytical processing) systems commonly process workloads in batches.
        - Gist: In batch processing, data is read and written in large, continuous chunks on disk (sequentially), which is faster than random access. Instead of many small operations, the system groups multiple read/write queries—such as aggregations, joins, or filters—into a single large job (like MapReduce), processing them together for high throughput but with higher latency.
    - Transactional processing:
        - (A.k.a online processing) follows an approach whereby data is processed interactively without delay, resulting in low-latency responses. 
        - Transaction workloads involve small amounts of data with random reads and writes and fewer joins.
        - OLTP (online transaction processing) system fall within this category.

#### **MapReduce**
- MapReduce is a widely used implementation of a batch processing framework.
- It is highly scalable and reliable and is based on the principle of divide-and-conquer, which provides built-in fault tolerance and redundancy.
- MapReduce does not require that the input data conform to any particular data model. Therefore, it can be used to process schema-less datasets.
- A dataset is broken down into multiple smaller parts, and operations are performed on each part independently and in parallel.
- The results from all operations are then summarized to arrive at the answer.
- Traditionally, data processing requires moving data from the storage node to the processing node that runs the data processing algorithm. This approach works fine for smaller datasets. However, with large datasets, moving data can incur more overhead than the actual processing of the data.
- With MapReduce, the data processing algorithm is instead moved to the nodes that store the data. The data processing algorithm executes in parallel on these nodes, thereby eliminating the need to move the data first. It saves network bandwidth and reduces processing time for large datasets.
- Terminology:
    - A MapReduce job is a unit of work that the client wants to be performed.
    - Hadoop runs the job by dividing it into tasks: map tasks and reduce tasks.
    - Hadoop divides the input to a MapReduce job into fixed-size splits.
    - Hadoop creates one map task for each split, which runs the user-defined map function for each record in the split.

<div class="row justify-content-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/mapreduce-1.png" title="mapreduce-1" class="img-fluid rounded z-depth-1" %}
    </div>
</div>￼

- Map:
    - The dataset is divided into multiple smaller splits. Each split contains multiple key-value pairs.
    - The map function (“mapper”) executes user-defined logic on each split: (K1, V1) -> list(K2, V2)
    - A mapper may generate zero(filtering) or more than one(demultiplexing) key-value pairs.
- Combine:
    - With larger datasets, moving data between map and reduce stages is more expensive than the actual processing.
    - The optional combiner function summarizes a mapper’s output before it gets processed by the reducer: (K2, list(V2)) -> list(K2, V2)
- Partition:
    - If more than one reducer is involved, a partitioning function (“partitioner”) divides the output from the mapper or combiner (if used) into partitions between reducer instances.
    - Although each partition contains multiple key-value pairs, all records for a particular key are assigned to the same partition.
- Shuffle & Sort:
    - Output from all partitioners is copied across the network to the nodes running the reduce task.
    - The key-value pairs are grouped and sorted according to the keys, list(K2, V2) -> (K2, list(V2))
- Reduce:
    - The reduce function (“reducer”) further summarizes its input: (K2, list(V2)) -> list(K3, V3).
    - The output of the reducer is then written as a separate file. One file per reducer. 
  
<div class="row justify-content-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/mapreduce-2.png" title="mapreduce-2" class="img-fluid rounded z-depth-1" %}
    </div>
</div>￼

- MapReduce in action (Ex: NCDC weather dataset)
    - Note: In Hadoop’s TextInputFormat, each line of a file is given a key representing its byte offset from the start of the entire file, not just the split. Offsets are used instead of line numbers because files are split and processed in parallel across multiple mappers, and byte offsets allow each mapper to locate its data efficiently without reading previous lines, ensuring scalability and consistency.
    - Note: Line numbers can’t be used in Hadoop because the input files are split and processed in parallel by multiple mappers. Each mapper starts reading from a different byte position in the file, so it has no way to know how many lines came before its split without scanning the entire file sequentially. Byte offsets, on the other hand, can be determined directly from the file’s position on disk, making them independent, efficient, and uniquely identifiable across splits — perfect for distributed processing.
    - The map function extracts the year and the air temperature, and emits them as its output.
    - The output from the map function is processed by the MapReduce framework before being sent to the reduce function. This processing sorts and groups the key-value pairs by key.
    - For each input, the reduce function iterates through the list and picks up the maximum reading
￼
<div class="row justify-content-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/mapreduce-3.png" title="mapreduce-3" class="img-fluid rounded z-depth-1" %}
    </div>
</div>￼

<div class="row justify-content-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/mapreduce-4.png" title="mapreduce-4" class="img-fluid rounded z-depth-1" %}
    </div>
</div>￼
￼
<div class="row justify-content-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/mapreduce-5.png" title="mapreduce-5" class="img-fluid rounded z-depth-1" %}
    </div>
</div>￼

<div class="row justify-content-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/mapreduce-6.png" title="mapreduce-6" class="img-fluid rounded z-depth-1" %}
    </div>
</div>￼
￼
- Data flow
    - Where to run the map task?
    - Data locality: run the map task on a node where the input data resides in HDFS, because it doesn’t use valuable cluster bandwidth.
    - If not possible, the job scheduler will look for a free map slot on a node in the same rack as one of the blocks, and the required data block is transferred over the rack’s local network.
    - If still not possible, an oﬀ-rack node is used, which results in an inter-rack network transfer.
    - So, what’s transferred is the actual HDFS block data required by the Map task to perform its computation.
￼
<div class="row justify-content-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/mapreduce-7.png" title="mapreduce-7" class="img-fluid rounded z-depth-1" %}
    </div>
</div>￼
<!-- fdaf -->

- How large is an input split?
    - Map tasks process input splits in parallel. So the processing is better load balanced when the splits are small.
    - However, if splits are too small, the overhead of managing the splits and map task creation begins to dominate the total job execution time.
    - For most jobs, a good split size tends to be the size of an HDFS block, which is 128 MB by default. It is the largest size of input that can be guaranteed to be stored on a single node.
- Where to store the map output?
    - Map tasks write their output to the local disk, not to HDFS. Why? Map output is intermediate output. It’s processed by reduce tasks to produce the final output. Once the job is complete, the map output can be thrown away.
    - Storing it in HDFS with replication would be overkill.
    - What if the node running the map task fails before the map output has been consumed by the reduce task? If a node fails before the reduce phase reads its map output, Hadoop simply re-runs the failed map task on another node. Since the map output is intermediate and deterministic, it can be regenerated from the original input data stored safely in HDFS.
- Why doesn’t Hadoop move the map task to another node that holds the same HDFS block instead of transferring the data? Why move data instead of computation? 
  - A: Hadoop’s scheduler tries first to move computation (the map task) to where the data already resides — this is called data locality, and it’s the preferred option. However, if all nodes holding that block are busy (no free map slots), the scheduler can’t wait indefinitely because it would delay the job. In that case, it assigns the task to another available node and transfers the required data block over the network. So, Hadoop only moves data as a fallback when moving computation isn’t immediately possible, balancing performance and cluster utilization.
- Do reduce tasks enjoy data locality?
- The input to each reduce task is normally the output from all mappers.
￼
<div class="row justify-content-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/mapreduce-8.png" title="mapreduce-8" class="img-fluid rounded z-depth-1" %}
    </div>
</div>￼

- Minimizing the data transferred between map and reduce tasks
    - The combiner function is an optimization that runs on the map output to reduce the amount of data transferred to the reducers by performing local aggregation. Hadoop does not provide a guarantee of how many times it will call it for a particular map output record. Calling the combiner function zero, one, or many times should produce the same output from the reducer.
    - It works best for associative and commutative operations, where partial results can be safely combined.
    - Which of the following data processing can benefit from a combiner function? 
        - Count the number of occurrences. YES
        - Find the maximum value. YES
        - Find the average value. NO, since the combiner may process subsets differently, leading to incorrect results unless additional logic (like tracking sums and counts separately) is used.
        - Filter values based on a predicate. YES

#### **Architecture**

**YARN (Yet Another Resource Negotiator)**

- YARN is Hadoop’s cluster resource management system.

<div class="row justify-content-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/mapreduce-9.png" title="mapreduce-9" class="img-fluid rounded z-depth-1" %}
    </div>
</div>￼
￼
- YARN provides its core services via two types of long-running daemons.
- Resource manager (only one in the cluster): manage the use of resources across the cluster.
- Node managers (on each node): launch and monitor containers. A container executes an application-specific process with a constrained set of resources (memory, CPU, …).
￼
<div class="row justify-content-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/mapreduce-10.png" title="mapreduce-10" class="img-fluid rounded z-depth-1" %}
    </div>
</div>￼

- Running an application: A client contacts the resource manager and asks it to run an application master process. The resource manager finds a node manager that can launch the application master in a container. The application master may request more containers from the resource manager. The application master use them to run a distributed computation (e.g., MapReduce).
- Types of YARN scheduler: (FIFO, Capacity, FAIR) scheduler.

<div class="row justify-content-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/mapreduce-11.png" title="mapreduce-11" class="img-fluid rounded z-depth-1" %}
    </div>
</div>￼￼

**Hadoop**

- Running a MapReduce job:
- Client: submit the MapReduce job.
- YARN resource manager: coordinate the allocation of compute resources in cluster.
- YARN node managers: launch and monitor the containers on machines in the cluster.
- MapReduce application master: coordinate the tasks running the MR job. The application master and the MapReduce tasks run in containers scheduled by the resource manager and managed by the node managers.
- HDFS: share job files between the other entities.

<div class="row justify-content-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/mapreduce-12.png" title="mapreduce-12" class="img-fluid rounded z-depth-1" %}
    </div>
</div>￼￼

- Progress & status updates:
- A job and each of its tasks have a status.
- The state of the job or task (e.g., running, successfully completed, failed).
- The progress of maps and reduces.
- The values of the job’s counters.
- A status message or description. 

<div class="row justify-content-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/mapreduce-13.png" title="mapreduce-13" class="img-fluid rounded z-depth-1" %}
    </div>
</div>￼￼

- Q: Does each node hold HDFS blocks and containers for task execution, managed by the NodeManager?
- A: Yes. Each node stores blocks as part of HDFS and can also run containers, which execute tasks (Map or Reduce) scheduled by YARN. The NodeManager on each node handles launching, monitoring, and reporting on these containers, while the ResourceManager coordinates cluster-wide resource allocation.
- Q: What does the YARN scheduler do? Does it schedule Map and Reduce tasks, and how is memory/CPU utilization handled?
- A: The YARN scheduler (FIFO, Capacity, or FAIR) manages the allocation of resources—CPU, memory, and containers—across the cluster. It decides which nodes get containers for tasks. While it schedules containers for Map and Reduce tasks indirectly via the ApplicationMaster, the actual memory and CPU usage is constrained per container on individual nodes as specified by the scheduler.
- Q: Does the ResourceManager use HDFS block information to improve data locality and distributed efficiency?
- A: Yes. The ResourceManager, through the ApplicationMaster, tries to schedule tasks on nodes that already hold the input HDFS blocks to exploit data locality, reducing network transfer and improving performance. If local nodes aren’t available, it may schedule tasks on the same rack or another node as a fallback.
- Q: What are job counters in Hadoop, and what do they mean?
- A: Counters are metrics collected during job execution. They track things like the number of bytes read/written, number of records processed, map/reduce task attempts, and custom user-defined counters. Counters provide insight into job progress, efficiency, and can help debug or optimize jobs.
- Q: What is a status message, and how is it used?
- A: A status message is a short description of the current state of a job or task (e.g., “Reading input”, “Merging outputs”, “Task failed”). It helps users and administrators monitor progress, understand failures, and debug issues during job execution.
- Gist: In Hadoop with YARN, each node stores HDFS blocks and also runs containers for executing tasks, with the NodeManager handling container lifecycle and reporting. The ResourceManager coordinates cluster-wide resource allocation, while the YARN scheduler (FIFO, Capacity, or FAIR) decides which nodes get containers, controlling CPU and memory usage per container. To maximize efficiency, tasks are ideally scheduled on nodes holding the relevant HDFS blocks, leveraging data locality; if unavailable, tasks may run on the same rack or another node. During execution, job counters track metrics like bytes read/written, records processed, and task attempts, providing insight for monitoring and optimization. Status messages report the current state of jobs or tasks, helping users and administrators monitor progress and debug issues.

**Resilience**

- Where can a failure happen? It can happen in 4 places: A MapReduce task, the MapReduce application master, YARN node manager or the YARN resource manager.
- Task failure:
    - Possible causes: Mapper/reducer bug, JVM bug, Hanging tasks: timeout (default: 10 min) exceeded without a progress update.
    - The application master will reschedule the task on another node manager.
    - If a task fails too many times (default: 4), it will not be retried again, and the whole job will fail.
- Application master failure:
    - An application master sends periodic heartbeats to the resource manager. 
    - In the event of application master failure, the resource manager will detect the failure and start a new instance of the master running in a new container (managed by a node manager).
    - In the case of the MapReduce application master, it’ll use the job history to recover the state of the tasks that were already run by the (failed) application, so they don’t have to be rerun.
    - If a MapReduce application master fails too many times (default: 2), it will not be retried again, and the whole job will fail.
    - Q: Where is the MapReduce job history stored and retrieved if the application master fails?
    - A: The job history is persisted in HDFS, not in the application master’s memory. When a new application master is started after a failure, it reads the job history from HDFS to recover the state of already completed or partially completed tasks, so they don’t have to be rerun.
- Node manager failure:
    - The resource manager will notice a node manager that has stopped sending heartbeats (default: 10 min) and remove it from its pool of nodes to schedule containers on.
    - Q: If a NodeManager fails, do we need to rerun completed Map tasks?
    - Yes, because the intermediate results are only stored in the node’s disk and not in hdfs. So, we’ve to rerun, so as to collect all the intermediate results from all mappers before the reducer part.
    - Q: If a NodeManager fails, do we need to rerun completed Reduce tasks?
    - A: No. Completed Reduce tasks write their output to HDFS, which is replicated and durable. Only in-progress Reduce tasks on the failed node need to be rescheduled on another node.
- Resource manager failure:
    - Failure of the resource manager is serious, because without it, neither jobs nor task containers can be launched. 
    - In the default configuration, the resource manager is a single point of failure, since in the (unlikely) event of machine failure, all running jobs fails, and can’t be recovered. 
    - For high availability (HA), we need to configure a standby resource manager.
    - Q: In ResourceManager failure, do we need to store information about all running applications?A: Yes. To recover after a ResourceManager failure, information about all running applications—including job metadata and application master states—needs to be persisted(not in HDFS). In HA setups, a standby ResourceManager keeps this information in ZooKeeper or shared storage(NFS) to resume operations without losing job information.
    - Q: In ResourceManager failure, do we need to store NodeManager information?A: No. It can be reconstructed back again from the NodeManager’s heartbeats.
    - Q: In ResourceManager failure, what about task information?A: Task information (status, progress, container allocation) is primarily tracked by the ApplicationMaster. The ResourceManager coordinates resources but doesn’t store detailed task outputs. In HA setups, the standby ResourceManager works with running ApplicationMasters to continue scheduling containers and managing tasks without losing track of progress.
- Shuffle and sort (Most important part)
    - MapReduce guarantees that the input to every reducer is sorted by key.
    - Flow: In MapReduce, each map task processes an input split, which contains multiple records. The map function runs on each record and generates intermediate key-value pairs, which are initially buffered in memory rather than written to disk immediately. This in-memory buffering allows the system to perform local partitioning, which determines which reducer each key belongs to, as well as optional combining, which reduces the number of intermediate records by aggregating data locally before transfer. Additionally, the intermediate data is sorted in memory by key to facilitate efficient merging later. Once the buffer reaches a configured threshold (default 100 MB), it is spilled to disk as a temporary file. Large map outputs may generate multiple spill files, which are later merged using a merge sort into a single sorted file per partition. During the shuffle phase, reducers fetch the intermediate data from all mappers, ensuring that each reducer receives all data corresponding to its assigned partition. Because the map output is already sorted locally, reducers perform a multi-way merge rather than a full sort, which improves efficiency. The final merged data is then streamed directly to the reduce function without writing it back to disk, minimizing I/O. If the intermediate data is too large to fit in memory, MapReduce applies external sorting, reading chunks of data from disk, merging them in memory, and writing back to disk. After merging, the reducer processes the records using the reduce function and writes the final output to HDFS. Throughout this process, combiners may optionally reduce the volume of data transferred, merges are optimized with multi-way strategies, and copying of map outputs can begin even before all map tasks finish, though the reduce function only runs after all intermediate data for that partition has been fetched. 

<div class="row justify-content-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/mapreduce-14.png" title="mapreduce-14" class="img-fluid rounded z-depth-1" %}
    </div>
</div>￼￼

**Speculative execution**

- The job execution time is sensitive to slow-running tasks (“stragglers”).
- Hadoop doesn’t try to diagnose and fix slow-running tasks; instead, it tries to detect when a task is running slower than expected and launches another equivalent task as a backup.
- The scheduler tracks the progress of all tasks of the same type (map and reduce) in a job, and only launches speculative duplicates for the small portion that are running significantly slower than the average.
- When a task completes successfully, any duplicate tasks that are running are killed since they are no longer needed. 
- Q: Which part of Hadoop handles speculative execution?
- A: Speculative execution is managed by the Hadoop MapReduce framework itself, specifically by the ApplicationMaster in YARN (Hadoop 2+).
- Q: Is the YARN scheduler part of the ApplicationMaster, and how do they interact?
- A: No, the YARN scheduler is part of the ResourceManager and operates cluster-wide, deciding which nodes get containers based on available resources and scheduling policies (FIFO, Capacity, FAIR). It does not manage task-level details like progress or stragglers. The ApplicationMaster, on the other hand, is job-specific. It requests containers from the ResourceManager (which allocates them via the scheduler), monitors task progress, handles failures, and manages speculative execution. Essentially, the ApplicationMaster depends on the scheduler for container allocation, and once allocated, it schedules and manages tasks within those containers, coordinating retries, merges, and data flow. The scheduler handles resource allocation, while the ApplicationMaster handles job and task management.