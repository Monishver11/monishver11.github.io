---
layout: post
title: HBase
date: 2025-12-16 16:36:00-0400
featured: false
description: Realtime and Big Data Analytics Course at NYU Courant - Conceptual Notes 7
tags: 
categories: RBDA-NYU
giscus_comments: true
related_posts: false

---

#### **OLTP (online transactional processing)**

- OLTP enables the real-time execution of large numbers of database transactions by large numbers of people, typically over the internet.
- A database transaction is a change, insertion, deletion, or query of data in a
database.
- In OLTP, the common, defining characteristic of any database transaction is its atomicity—a transaction either succeeds as a whole or fails (or is canceled). It cannot remain in a pending or intermediate state.
- Characteristics of OLTP systems
  - Process a large number of relatively simple transactions.
  - Enable multi-user access to the same data, while ensuring data integrity.
  - Emphasize very rapid processing, with response times measured in milliseconds.
  - Provide indexed datasets.
  - Are available 24/7/365.
- OLTP versus OLAP

<div class="row justify-content-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/hbase-1.png" title="hbase-1" class="img-fluid rounded z-depth-1" %}
    </div>
</div>￼

#### **NoSQL (Not only SQL) databases**

- NoSQL refers to Not only SQL databases.
- They are non-relational databases for storing huge datasets eﬀectively.
- They are good for:
  - Indexing huge amount of documents.
  - Serving pages on high-traﬃc websites.
  - Delivering streaming media.
- Consistency is less important. ACID properties are traded for performance.
- NoSQL refers to a range of databases that are not relational databases.
  - They do not support SQL.
  - They often do not guarantee ACID properties.  
- Many NoSQL databases are descendants of Google’s Bigtable and Amazon’s Dynamo.
  - They are designed to be distributed across many nodes.
  - They usually provide eventual consistency.
  - They have very flexible schema.
- NoSQL databases do not use SQL for data manipulation.
  - The database is optimized for retrieval and append operations.
  - They oﬀer a key-value store. Some are built on GFS/HDFS.
- NoSQL databases are designed for scalability and performance.
  - They are useful for big data in applications where a relational model is not needed. 
  - They can easily scale up by adding inexpensive commodity servers. Much easier than with relational databases.
- NoSQL database systems are developed to manage large volumes of data that do not necessarily follow a fixed schema.
  - Data is sharded and stored across many servers.
  - The architecture is distributed and fault-tolerant. 
  - They are useful for managing large amounts of data where satisfying realtime constraints is the priority. The goal is near-realtime or soft realtime, i.e., fast enough for a web service.
- Q: Do NoSQL databases relax consistency but keep atomicity?
- A: In most NoSQL databases, atomicity is preserved but only at a limited scope, such as a single row, document, or key. Consistency across multiple records is often relaxed, and isolation guarantees are weaker, while durability is usually maintained. This trade-off is intentional, allowing NoSQL systems to achieve high scalability and performance in distributed environments.
- Google’s solution for NoSQL
  - Bigtable: a distributed storage system for structured data
  - Best Paper award at OSDI’06 (one of the two most prestigious systems conferences held once every two years).
  - As a side note, Google also published Chubby in the same conference. 
  - Today, Bigtable is still one of the most widely-used NoSQL databases. Google oﬀers Cloud Bigtable as part of its cloud computing services.
- Hadoop’s solution for NoSQL
  - HBase is modeled after Google’s Bigtable.
  - Bigtable is closed-source; HBase is open-source.
  - It is suitable for extremely large databases. Billions of rows, millions of columns.
  - It is distributed across thousands of nodes.
  - Facebook used HBase for its messaging system from 2010 to 2018.

#### **HBase**

- **Use cases**
  - Facebook messages: At the high end: over one million HBase cluster operations per second. Processing thousands of records per second per node.
  - Large amount of stored data: Queries only require small amount of rows in response.
  - Use HBase for random reading or writing, or both: When your data is in the TB range.

- **NoJOIN?**
  - Traditional JOIN operations are not supported in NoSQL.
  - Implementing JOIN is impractical.
    - In HBase, data is sharded across many servers.
    - HBase wants to provide fast response.
  - But in HBase, the capability exists for very, very large rows (millions of columns).
    - JOINs are not needed.
    - The recommendation is that data should be de-normalized.
  - Conceptual Flow: 
    - HBase does not support traditional SQL JOIN operations because implementing joins in a distributed NoSQL system is impractical and expensive. In HBase, data is sharded across many servers, so performing a join would require large amounts of network communication and coordination, which would significantly increase latency. Since HBase is designed for fast, low-latency access, joins would violate its performance goals. Instead, HBase supports very wide rows with potentially millions of columns, allowing related data to be stored together in a single row. Because of this design, joins are usually unnecessary, and the recommended approach is to denormalize the data, storing all frequently accessed related data together to enable fast reads. 
    - In HBase, data is sharded (partitioned) by row key, not by columns. Each row can be extremely wide (even millions of columns), and all columns for a given row are stored together. HBase uses the row key to determine which region (and therefore which server/node) holds that row. 
    - So when a request comes in, HBase routes it directly to the node responsible for that row key, allowing fast reads and writes without needing joins. This row-based sharding, combined with wide rows and denormalized data, is what enables HBase to scale horizontally while maintaining low-latency access.

- **Sparse rows**
  - HBase is a wide-column store.
  - HBase can handle sparse records.
    - Sparse records means that some columns are not filled in.
  - In HBase, there is no penalty for sparse data because no space is allocated. This is in contrast to relational databases in which an unpopulated field is also allocated space.

- **Designing tables**
  - The RDBMS approach to design is relationship-centric.
  - However, HBase requires an access-centric approach to design. 
  - We cannot take an RDBMS and model it directly in HBase.
    - RDBMS will have normalized data. However, HBase will perform better with de-normalized data.
    - HBase columns must be grouped into column families based on expected access patterns.
    - The columns of a column family are stored close together on disk for fast access.
  - We'll see more on this in HBase's Data Model (see below).

- **NoACID?**
  - HBase is not a fully ACID-compliant database.
    - HBase does not provide strong consistency.
    - It does not provide atomicity across rows.
    - However, it does provide atomic operations at the row level.

- **Storage platform**
  - HBase uses HDFS underneath for storage.
  - HBase does not convert commands to MapReduce jobs.
  - HBase supports random reads and writes.
    - Writes are implemented through versioning of cells.
    - The user can define the max number of versions to maintain.
    - The user can also define the time-to-live. After that time, the row is automatically marked for deletion by HBase.


#### **HBase Architecture**

<div class="row justify-content-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/hbase-2.png" title="hbase-2" class="img-fluid rounded z-depth-1" %}
    </div>
</div>￼

- An HBase cluster is comprised of master and worker nodes.
- Similar master-worker architecture as ween with…
  - HDFS: NameNode and DataNodes.
  - YARN: ResourceManager and NodeManagers.
  - MapReduce: ApplicationMaster and tasks.
  - Trino: Coordinator and workers.
- **Master node**
  - The master node manages a cluster of Regionservers.
  - It bootstraps the initial install.
  - It assigns regions to registered Regionservers.
  - It recovers Regionserver failures.
- The master node is lightly loaded.
  - Client data does not move through the master node.
  - Clients do not rely on the master node for region location information.

- **Regionservers**
  - Each Regionserver carries zero or more regions.
    - A region is a subset of a table’s rows.
    - Row updates are atomic.
  - They handle read/write requests.
  - They perform region splits, which they communicate to the master node. 
    - Writes cause regions to grow. Eventually, they must be split.

- **ZooKeeper cluster**
  - ZooKeeper is a distributed coordination service (e.g., configuration, synchronization, naming registry).
  - HBase uses ZooKeeper to host vitals such as:
    - The location of the hbase:meta catalog table.
    - The address of the current cluster master.
  - HBase also uses ZooKeeper to host the transaction state of region assignments. This is to support fast recovery.
  - More clearly: In HBase, the transaction state of region assignments stored in ZooKeeper represents the current and in-progress states of assigning regions to RegionServers, such as unassigned, assigning, or assigned. ZooKeeper maintains this shared state so the HBase master and RegionServers have a consistent view of region ownership. If a master or RegionServer fails during an assignment, the new master can read this state from ZooKeeper and safely resume or recover the operation, ensuring that each region is assigned to exactly one RegionServer at a time and enabling fast, reliable cluster recovery.
  - Fresh clients connect to the ZooKeeper to learn the location of hbase:meta. The result is cached at the clients until there is a fault.

- **The hbase:meta catalog table**
  - It maintains the current list, state, and locations of all user-space regions afloat on the cluster.
  - Entries in hbase:meta are keyed by region name, which is made up of:
    - The table name.
    - The start row key.
    - The creation time.
    - A checksum.
  - Conceptual Flow:
    - The hbase:meta catalog table is a special system table in HBase that acts as the central directory for all user-space regions in the cluster. It keeps track of the current list of regions, their states (e.g., online, offline, splitting), and the RegionServers that host them. Each entry in hbase:meta is keyed by a region name, which is a combination of the table name, the region’s start row key, the creation timestamp, and a checksum to ensure uniqueness.
    - This catalog table is essential for HBase’s operation: when a client wants to read or write data, it queries hbase:meta (directly or via ZooKeeper) to determine which RegionServer holds the desired row. Similarly, the HBase master uses it to manage region assignments, splits, and recovery. By maintaining an up-to-date map of the cluster’s regions, hbase:meta enables fast lookups, load balancing, and fault-tolerant access to data in a distributed environment.

- **Cluster expansion**
  - Expanding an HBase cluster is much easier than scaling a traditional relational database. 
  - In RDBMS systems, expansion is often complex, error-prone, and difficult to maintain, because horizontal scaling was not part of their original design. Features like joins, complex queries, and strict ACID guarantees make distributing data across multiple nodes challenging. 
  - In contrast, HBase is designed for horizontal scalability, allowing nodes to be added with minimal disruption. Its row-based sharding, denormalized data models, and simplified consistency guarantees make it straightforward to expand the cluster and handle growing datasets efficiently.

#### **HBase Data model**    

<div class="row justify-content-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/hbase-3.png" title="hbase-3" class="img-fluid rounded z-depth-1" %}
    </div>
</div>￼

- **Column family**
  - HBase is a distributed column-family-oriented database built on top of HDFS.
  - A column family is a grouping of columns.
    - Columns that belong to a given column family are physically stored together.
    - A column is referenced as columnFamilyName:columnName. The columnName is sometimes referred to as the qualifier.
    - In HBase, tuning is performed on a column family bases.
  - HBase column families are defined at table definition time, but columns can be defined dynamically.
  - Conceptual Flow:
    - In HBase, a column family is a grouping of columns whose data is physically stored together in HDFS files called HFiles, allowing efficient access to all columns in the family. 
    - Columns are referenced as columnFamily:columnName, where columnName is also called the qualifier. Column families are defined at table creation, but individual columns within a family can be added dynamically. 
    - All regions of a table share the same column family structure, even though many rows may be sparse and lack values for some columns; empty cells are simply not stored. 
    - HFiles are stored on HDFS and subject to the HDFS block size (typically 128 MB); if a column family’s data exceeds a block, it is automatically split across multiple blocks distributed across different data nodes. HBase manages reading and writing from these HFiles, including handling multiple blocks, region splits, and region-to-node assignments, using metadata and indexing to ensure fast, atomic access at the row level. 
    - This design supports very wide tables, sparse and denormalized data, and scalable, high-throughput access while keeping related columns physically close for efficient reads.

- **Cells**
  - Values are stored in HBase cells.
  - A cell lies at the intersection of a row and a column, at a particular version.
  - By default, a cell version is a timestamp. The timestamp is automatically assigned by HBase.
  - To reference a value in a cell, use: row key + columnFamily:columnName + timestamp.
  - A cell’s content is an uninterpreted array of bytes.
- Regions
  - A region (“tablet” in Google’s jargon) is a subset of an HBase table’s rows.
  - It is defined by a start row key (inclusive) and an end row key (exclusive). Every table’s rows belong to some HBase region.
  - Initially, by default, a table is comprised of just one region. As the size of the region grows, it splits.
  - A region splits at a row boundary into two new regions of about equal size. The threshold at which a region will be split is configurable. 
  - In HBase, when a region splits at a row boundary, it means the split happens between two rows, not in the middle of a row’s data. Each row is the atomic unit for storage and updates, so HBase ensures that a row’s contents are never divided across two regions.
  - Regions are the units that get distributed throughout an HBase cluster. This is how a table can grow very large without the constraints that hamper RDBMS table grows.
  - A table’s total content is the full set of the regions that hold its rows.
- Row keys
  - In HBase, an operation on a given row is atomic. So, the locking model is simple.
  - HBase provides just one index: the row key.
  - Row keys are stored in sorted order. It is fast and easy to locate a particular row via lookup. Each row has a row key. The row key is also an array of bytes.

<div class="row justify-content-center">
  <div class="col-sm-7 mt-3 mt-md-0">
      {% include figure.liquid path="assets/img/hbase-4.png" title="hbase-4" class="img-fluid rounded z-depth-1" %}
  </div>
</div>￼

- **Write operation**
  - On the Regionserver, writes are assisted by the Write Ahead Log (WAL).
  - All writes are first written to the WAL on HDFS, and then to the memstore (Google calls it the memtable) for quick lookup.
  - When the amount of data in the memstore reaches a threshold (configurable), the memstore is flushed to HDFS.
  - Each time data is flushed from the memstore, it is stored on disk in an HFile (Google uses SSTable).
- Read operation
  - The region’s memstore is first consulted. If more versions are needed, flush files are consulted from newest to oldest.
  - Reads are assisted by the block cache: The block cache is in the Regionserver. It uses the Least Recently Used (LRU) algorithm.
  - The cache is configurable: It can model multi-level cache. It can choose where data is cached.
  - So while the cache is local to each RegionServer, you can tune it independently on different servers to optimize read performance based on workload and memory availability.
- Delete operation
  - When you delete a version, it means to delete all cells where the version is less than or equal to this version.
  - Since HBase never modifies data in place, it will not immediately delete (or mark as deleted) the entries in the HFiles.
  - Instead, it writes a tombstone marker, which will mask the deleted values.

- **Compactions**
  - Minor compaction is a process that compacts a (configurable) number of adjacent small HFiles into a large HFile. It does not drop deletes or expired versions.
  - Major compaction is a heavyweight process that: Rewrites all files within a column family for a region into a single new file. Removes tombstone markers any dead entries. Deletes any expired data (Based on TTL).

- **Note:**
  - Q: What is the relationship between HFiles and HDFS blocks in HBase?
  - A: HFiles are logical files managed by HBase that store the data for a column family. They are physically stored on HDFS, which splits files into blocks (typically 128 MB). If an HFile is smaller than the HDFS block size, it may share a block with other small files, but each HFile is usually treated as a separate file. When an HFile grows larger than the block size, HDFS splits it across multiple blocks stored on different data nodes. HBase manages these HFiles independently, using indexing and metadata for efficient reads, while HDFS handles their block-level storage and distribution.
  - So, HFiles are a logical abstraction created by HBase to manage data storage on HDFS. They enable HBase to efficiently store column-family data, maintain indexes, metadata, and Bloom filters for fast reads, and handle memstore flushes, compactions, and region splits. Physically, HFiles are just normal HDFS files, but the abstraction allows HBase to manage wide rows, sparse data, multiple versions, and efficient retrieval without exposing the underlying HDFS complexities to users.
  - Conceptual flow of how HBase routes read and write requests to the correct RegionServer: When a client wants to read or write a row, it first contacts ZooKeeper to find the location of the hbase:meta catalog table. The client then queries hbase:meta to determine which RegionServer hosts the region containing the target row key. With this information, the client communicates directly with the appropriate RegionServer, which manages the region’s memstore and HFiles to handle the read or write. The client caches the region-to-RegionServer mapping for future requests to avoid repeated lookups. If a region moves due to splits or server failures, the cache is updated using ZooKeeper and hbase:meta. Throughout this process, the HBase master is not in the path of reads or writes; it only coordinates region assignments, splits, and recovery, while RegionServers perform the actual data storage and access.

#### **HBase Usage**

- **Programming**
  - Aside from the HBase shell, you can use compiled languages with the HBase API.
  - Java is natively supported.
  - Python is also supported with the Thrift server. Python will be slower than Java due to the Thrift server overhead (additional interface).
  - Multiple Thrift servers may be needed for improved performance when datasets are large.

- **MapReduce**
  - HBase can be used as a source and/or sink in MapReduce jobs.
  - The TableInputFormat class makes splits on region boundaries so maps are handed a single region to work on.
  - The TableOutputFormat class will write the result of the reduce into HBase.

#### **Summary**

- HBase is a distributed column-oriented database built on top of HDFS.
- No real indexes: rows and columns are stored sequentially.
- Automatic partitioning: Regions are split and distributed across the cluster.
- Scale linearly with new nodes: Regions automatically rebalance.
- Commodity hardware: much less I/O hungry than RDBMSs.
- Fault tolerance: no need to worry about individual node downtime.
- Batch processing: support distributed MapReduce jobs with locality awareness.

---

**Doubts:**
- In Webtable example(Slide 31), What is the reason for having the row-key as “com:cnn.www”? Understand it clearly.