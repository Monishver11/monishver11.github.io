---
layout: post
title: Hive & Trino
date: 2025-12-15 23:40:00-0400
featured: false
description: Realtime and Big Data Analytics Course at NYU Courant - Conceptual Notes 6
tags: 
categories: RBDA-NYU
giscus_comments: true
related_posts: false

---

#### **Online Analytical Processing (OLAP)**
- OLAP is software for performing multidimensional analysis at high speeds on large volumes of data from a data warehouse, data mart, or some other unified, centralized data store.
- Most business data have multiple dimensions—multiple categories into which the data are broken down for presentation, tracking, or analysis.
- But in a data warehouse, data sets are stored in tables, each of which can organize data into just two of these dimensions at a time. OLAP extracts data from multiple relational data sets and reorganizes it into a multidimensional format that enables very fast processing and very insightful analysis.
- Example: Sales figures might have several dimensions related to Location (region, country, state/province, store), Time (year, month, week, day), Product (clothing, men/women/children, brand, type);

<div class="row justify-content-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/hive-1.png" title="hive-1" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
  
#### **Hive**

- Hive is a data warehousing framework developed by Facebook.
- Hive is built on top of HDFS (for storage) and MapReduce (for processing).
- Hive supports HiveQL:
    - HiveQL is a dialect of SQL.
    - HiveQL is heavily influenced by MySQL.
- User data are organized into tables, which are stored as files on HDFS.

- **Metastore**
    - It’s the central repository of Hive metadata such as table schemas.
    - It’s a relational database automatically created by Hive.
- Q: What is stored in the Hive Metastore?
- A: The Hive Metastore stores metadata about Hive objects, not the actual data. This includes database and table definitions, column names and data types, partition information and locations, storage formats and SerDe details, table properties, statistics used by the query optimizer, and access privileges. The actual table data itself is stored separately in HDFS.
- Q: Why does Hive use an RDBMS for the Metastore?
- A: Hive uses an RDBMS for the Metastore because metadata is highly structured and relational, requiring fast, indexed lookups and strong consistency. An RDBMS provides ACID guarantees and supports concurrent access from multiple clients, ensuring metadata remains consistent and durable. It is also optimized for frequent small reads and writes, which is inefficient on HDFS, making an RDBMS the right choice for storing Hive metadata.

- Hive converts your query into a set of MapReduce jobs. Therefore:
    - It’s batch-oriented.
    - Its response time is relatively long (tens of seconds to minutes).
- Hive provides a shell for interactively issuing commands.
- There are a number of Hive services in addition to the Hive shell.
- For example, the hiveserver2 service exposes a Thrift service
    - It enables access by clients written in diﬀerent languages.
    - It supports applications using Thrift, JDBC, ODBC connectors.
- RPC (Remote Procedure Call) is a communication mechanism that allows a program to execute a function or procedure on another machine or process as if it were a local function call. The client sends a request with the function name and arguments over the network, the server executes the function, and the result is sent back to the client. RPC abstracts away network details like sockets and message passing, making distributed systems easier to build.
- Thrift lets a service provide a set of functions (APIs) that can be called remotely, and programs written in different programming languages can use those functions without worrying about the details of how the network communication works. For example, HiveServer2 can run in Java, but a Python or C++ client can still send queries to it using Thrift. Essentially, Thrift handles the translation between languages and the network communication for you.
- HiveServer2 also supports standard connectivity through JDBC (Java Database Connectivity) for Java applications and ODBC (Open Database Connectivity) for applications in various other languages, enabling external tools and BI applications to interact with Hive without using the Hive shell directly.
  
- **Architecture**
<div class="row justify-content-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/hive-2.png" title="hive-2" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


- **Schema on write versus schema on read**
    - Schema on write
        - Traditional databases uses “schema on write”.
        - Used by relational databases like MySQL.
        - Schema is enforced at load time.
        - If the data being loaded does not conform to the schema, the load fails.
        - Queries are faster because columns can be indexed and data are compressed at load time. However, loading takes a long time.
    - Schema on read
        - Hive uses “schema on read”.
        - Table’s schema is not enforced until a query is issued.
        - Makes for much faster loading.
        - Schema on read is more flexible.
        - Multiple schemas can be supported simultaneously.
        - A Hive table is essentially an HDFS directory containing one or more files that comprises the table.
        - Users may define multiple Hive table schemas for a given Hive table.


- **Features**
    - Older versions of Hive did not support updates due to limitations of HDFS.
    - Throughout the years, Hive has been supporting more and more usages.
    - Now, Hive supports INSERT INTO for adding rows to existing tables.
    - Hive also supports:
        - Indexes (as of version 0.7.0).
        - Primitive types as found in Java.
        - Java complex types ARRAY and MAP.
        - STRUCT type, which is a record type.


- **Tables** 
    - A Hive table is logically made up of the data being stored and the associated metadata describing the layout of the data in the table.
    - The data typically resides in HDFS. The metadata is stored in the metastore, which is a relational database.
    - Hive supports managed tables and external tables.
    - Managed tables
        - Hive moves the data into its warehouse directory.
        - When you DROP a managed table, its metadata and its data are deleted.
    - External tables
        - You control the creation and deletion of the data.
        - Hive refers to the data at an existing location outside the warehouse directory.
        - When you DROP an external table, only the metadata is deleted, and the data is untouched.
    - In Hive, the warehouse directory is the default location on HDFS where Hive stores the data files for managed tables. By default, this is usually something like /user/hive/warehouse/. When you create a managed table, Hive moves or stores the table’s data in this directory, and it manages the lifecycle of both the metadata and the data. For external tables, the data stays outside this directory, and Hive only keeps metadata pointing to its location.

- **Partitions**
    - Hive allows partitions to be defined.
    - It divides a table based on the value of a partition column (e.g., date).
    - Using partitions can make it faster to do queries on slices of the data.
    - Partitions are nested subdirectories of the table directory on HDFS.
    - If a table has a nested directory structure in HDFS but no partition column is defined in Hive, Hive will not treat those directories as partitions. The table will be considered unpartitioned, and all the data files under the table directory (including subdirectories) will be read as a single logical table. Queries won’t automatically skip directories, they’ll scan all data files, which can make queries slower.
    - While Creating Hive partitions, note that while the partitions are referenced in queries just as if they were fields, no data has been added to the table contents.
    - Ex: hive> CREATE TABLE logs (timestamp BIGINT, line STRING) PARTITIONED BY (theDate STRING, campus STRING);
    - The LOAD DATA command is used to populate a table with data. The source of the data is specified after the INPATH keyword.
    - If multiple files are found in the same directory, they are all part of the table.
    - You can issue SQL queries on the Hive table.
    - In the query, the partition column is treated just like a column internal to the table, even though values are not contained in the data files.
    - Gist: A Hive table is a logical abstraction over data stored in HDFS and can be either managed, where Hive owns and stores the data in its warehouse directory, or external, where the data lives outside Hive and only metadata is managed. A partition in a Hive table means a logical subdivision of the table’s rows based on the values of one or more partition columns (for example, date or campus). Physically, each partition maps to a subdirectory in HDFS, named using the partition column values (e.g., date=2025-12-15/campus=NYC/). All files within a partition directory belong to that partition, and multiple files are treated as one logical chunk of the table. Partition column values are not stored inside the data files; Hive infers them from the directory structure. Data is loaded into tables and partitions using LOAD DATA or external tools, and Hive automatically associates files with partitions based on their HDFS paths. When querying, partition columns behave like normal table columns, but Hive can use them for partition pruning, reading only the relevant directories instead of scanning the entire table, which significantly improves query performance.


- **Joins**
    - Hive supports conventional SQL joins: Inner joins, Left/right/full outer joins, and Left/right semi joins.
    - Remember replicated joins (map-side joins or broadcast join)? They are automatic in Hive.


- **Extensibility**
    - Hive is extensible via:
        - UDFs: user-defined functions -> Input one row, output one row.
        - UDAFs: user-defined aggregate functions -> Input multiple rows, output one row.
        - UDTFs: user-defined table-generating functions -> Input one row, output multiple rows — a table.


- **Presto & Trino**
    - Since 2008, Hive became widely used within Facebook for running analytics against data in HDFS on its very large Hadoop cluster.
    - Hive was not suitable for interactive queries at Facebook’s Scale.
    - By 2012, Facebook’s Hive data warehouse was 250 PB in size and needed to handle
    - hundreds of users issuing 10k+ queries each day.
    - Hive could not query other data sources. This means Hive was largely limited to querying data stored in HDFS (and Hive-managed tables) and was not designed to easily query multiple, heterogeneous data sources in a single query.
    - Hive either could not access these sources directly or required complex ingestion pipelines to first copy the data into HDFS. This made interactive analytics slow and inflexible.
    - In 2012, Facebook started to develop Presto from scratch to address the performance, scalability, and extensibility needs for analytics at Facebook.
    - In 2018, its creators left Facebook with their project and later renamed it Trino.

#### **Trino**
  
TODO: Add Notes for Trino;


