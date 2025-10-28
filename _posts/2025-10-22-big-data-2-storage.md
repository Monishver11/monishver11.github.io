---
layout: post
title: Big Data Storage
date: 2025-10-22 11:25:00-0400
featured: false
description: Notes on Big Data Storage - Concepts and Architecture
tags: RBDA
categories: RBDA-NYU
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

- “Big Data” is not new. Oil companies, telecommunications companies, and other data-centric industries have had huge datasets for a long time. 
- As storage capacity continues to expand, today’s “big” is tomorrow’s “small.”
- “Big Data” is when the size of the data itself becomes part of the problem.
- Why is Big Data a problem?
    - Where can I store my company’s ever-growing data?
    - How much is that going to cost?
    - How am I going to manage all the hardware and software?
    - Users are asking bigger questions - how can I provide compute power?
    - And, I/O speed is a problem too.
- One 10 TB hard disk drive (HDD) vs 10 TB of storage with 100 HDDs, which is better? 
- 10 TB of storage with 100 HDDs has an advantage of having one read/write head per drive. So, the whole disk read/write time reduces overall when compared to the whole disk read/write time of the single 10 TB HDD.
- In practice, multiple disk drives are installed in one server, sometimes as many as 12 or more. Each drive is 2 TB ~ 6 TB in size.
- It is important to match the speed of the drives to the processing power of the server, because the CPU can become the bottleneck.
- New problems:
    - How can I read the parts of my file simultaneously from multiple drives on multiple servers?
    - How do I even know where the pieces of my file are, if they’re stored on multiple servers?
- Before answering these questions, we need to know some of the big data storage concepts that are used and involved.
- Big Data analytics uses highly scalable distributed technologies and frameworks to analyze large volumes of data from diﬀerent sources.
- To store Big Data datasets, often in multiple copies, innovative storage strategies and technologies have been created to achieve cost-eﬀective and highly scalable storage solutions.
- We’ll introduce the following concepts: Clusters, distributed file systems, relational database management systems, NoSQL, sharding, replication, CAP theorem, ACID, and BASE.
- A cluster is a tightly coupled collection of servers (“nodes”).
- These servers usually have the same hardware specifications and are connected together via a network to work as a single unit.
- Each node in the cluster has its own dedicated resources, such as memory, a processor, and a hard drive.
- A cluster can execute a job by splitting it into small pieces (“tasks”) and distributing their execution onto diﬀerent computers that belong to the cluster.
- Clusters -> Racks -> Nodes(servers)
- Next is Distributed file systems (DFS).
- A file is the most basic unit of storage to store data.
- A file system (FS) is the method of organizing files on a storage device.
- A DFS is a file system that can store large files spread across the nodes of a cluster. E.g., Google File System (GFS), Hadoop Distributed File System (HDFS).
- Next is Relational database management systems (RDBMS).
- A RDBMS is a product that presents a view of data as a collection of rows and columns.
- SQL (structured query language) is used for querying and maintaining the database.
- A transaction symbolizes a unit of work performed against a database, and treated in a coherent and reliable way independent of other transactions. 
- Next is NoSQL.
- A Not-only SQL (NoSQL) database is a non-relational database that is highly scalable, fault-tolerant and specifically designed to house semi-structured and unstructured data. 
- E.g., Key-value store: Redis, Dynamo. Document store: MongoDB, CouchDB. Wide column store: Bigtable, HBase, Cassandra. Graph store: Pregel, Giraph.
- Next is Sharding.
- Sharding is the process of horizontally partitioning a large dataset into a collection of smaller, more manageable datasets (“shards”).
- Each shard is stored on a separate node, and is responsible for only the data stored on it.
- All shards share the same schema. They collectively represent the complete dataset.
- How does sharding work in practice?
    - Each shard can independently service reads and writes for the specific subset of data that it is responsible for. 
    - Depending on the query, data may need to be fetched from both shards.
- Benefits of sharding:
    - Sharding allows the distribution of processing loads across multiple nodes to achieve horizontal scalability.
    - Sharding provides partial tolerance toward failures. In case of a node failure, only data stored on that node is aﬀected.
- Concerns with sharding:
    - Queries requiring data from multiple shards will impose performance penalties.
    - To mitigate such performance issues, data locality keeps commonly accessed data co-located on a single shard. This idea leads to the concept of replication.
- Replication stores multiple copies of a dataset (“replicas”) on multiple nodes.
- There are two methods of replication: Master-slave replication and Peer-to-peer replication.
- Master-slave replication
    - All data is written to a master node.
    - Once saved, the data is replicated over to multiple slave nodes.
    - Write requests, including insert, update and delete, occur on the master node.
    - Read requests can be fulfilled by any slave node.
    - This is ideal for read intensive loads. Growing read demands can be managed by horizontal scaling to add more slave nodes.
    - The writes are consistent. All writes are coordinated by the master node. However, write performance will suffer as the amount of writes increases.
    - If the master node fails, reads are still possible via any of the slave nodes. But, writes are not supported until a master node is reestablished.
    - For recovery, we resurrect the master node from a backup or choose a new master node from the slave nodes.
    - There is a concern of read inconsistency. 
    - Ex: User A updates data. The data is copied over to Slave A by the Master. Before the data is copied over to Slave B, User B tries to read the data from Slave B, which results in an inconsistent read. The data will eventually become consistent when Slave B is updated by the Master.
    - There are other solutions as well, but we’ll see those as we go and later.
- Peer-to-peer replication
    - All nodes (“peers”) operate at the same level.
    - Each peer is equally capable of handling reads and writes. 
    - Each write is copied to all peers.
    - In this replication strategy, we might face both read and write inconsistency.
    - Read inconsistency: User A updates data. The data is copied over to Peer A and Peer B. Before the data is copied over to Peer C, User B tries to read the data from Peer C, resulting in an inconsistent read. The data will eventually be updated on Peer C, and the database will once again become consistent.
    - Write inconsistency: A simultaneous update of the same data may happen across multiple peers.
    - Strategies to resolve these:
        - Pessimistic concurrency is a proactive strategy. It uses locking to ensure that only one update to a record can occur at a time. However, this is detrimental to availability since the database record being updated remains unavailable until all locks are released. 
        - Based on what is held by the lock and who holds the lock, there are many different ways of achieving this strategy. There can be write locks (exclusive locks) and read locks(shared locks), and the unit of locking depends on the system and its users, it could be fine-grained like at a record level, or coarse-grained at a table level. And since its peer-to-peer, the lock can be managed by centralized lock manager (as a service by zookeeper) or distributed lock manager (using a distributed consensus protocol like Paxos, Raft) or other application-level ownership.
        - Optimistic concurrency is a reactive strategy that does not use locking. Instead, it allows inconsistency to occur with knowledge that eventually consistency will be achieved after all updates have propagated.
- Sharding vs replication
    - Actually, both sharding and replication can be used together. 
    - We can combine, sharding and master-slave replication, sharding and peer-to-peer replication or any other commendations.
    - Sharding and master-slave replication
  
<div class="row justify-content-center">
    <div class="col-sm-9 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/sharding_ms.png" title="sharding_ms" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

    - Sharding and peer-to-peer replication

<div class="row justify-content-center">
    <div class="col-sm-9 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/sharding_pp.png" title="sharding_pp" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

- CAP theorem
    - A distributed database system may wish to provide three guarantees.
    - Consistency: a read request from any node results in the same, most recently written data across multiple nodes.
    - Availability: a read/write request will always be acknowledged in the form of a success or a failure.
    - Partition tolerance: the database system can tolerate communication outages that split the cluster into multiple silos and can still service read/write requests.
    - CAP theorem states that a distributed database system can only provide two of the three properties. C+A+P is not possible.
    - Although communication outages are rare and temporary, partition tolerance (P) must be supported by a distributed database.
    - Therefore, CAP is generally a choice between choosing C+P or A+P.
    - Below explains how things can go wrong in each CAP combinations.
    - C + A (no Partition Tolerance): When a network partition happens, nodes can’t communicate to stay consistent. Since the system isn’t partition-tolerant, it must shut down or reject requests, leading to unavailability.
    - C + P (no Availability): During a partition, nodes stop serving requests until they can synchronize again to maintain consistency. This causes temporary unavailability, as the system pauses updates to avoid conflicts.
    - A + P (no Consistency): When a partition occurs, all nodes continue serving requests independently. Because they can’t coordinate, updates may diverge, producing inconsistent or stale data until the partition heals and replicas reconcile.
    - In summary: Choosing C + A means all nodes must stay in sync, but the system fails if a partition occurs. Choosing C + P ensures data consistency across partitions, but some nodes may become unavailable. Choosing A + P keeps the system running despite partitions, but nodes may serve inconsistent data until synchronization occurs.
- ACID
    - ACID is a traditional database design principle on transaction management.
    - It stands for Atomicity, Consistency, Isolation and Durability.
    - Traditional databases leverages pessimistic concurrency controls (i.e., locking) to provide ACID guarantees.
    - Atomicity ensures that all transactions will always succeed or fail completely. In other words, there are no partial transactions. 
    - Ex: A user attempts to update three records as a part of a transaction. Two records are successfully updated before the occurrence of an error. As a result, the database rolls back any partial eﬀects of the transaction and puts the system back to its prior state.
    - Consistency ensures that only data that conforms to the constraints of the database schema can be written to the database.
    - Ex: A user attempts to update the “amount” column of the table that is of type “float” with a value of type “varchar.” The database rejects this update because the value violates the constraint checks for the “amount” column.
    - Note: consistency here is different from the consistency in a distributed database system. 
    - Isolation ensures that the results of a transaction are not visible to other operations until it is complete.
    - Ex: User A attempts to update two records as part of a transaction. The database successfully updates the first record. However, before it can update the second record, User B attempts to update the same record. The database does not permit User B’s update until User A’s update succeeds or fails completely. This occurs because the record with id = 3 is locked by the database until the transaction is complete.
    - Durability ensures that the results of a transaction are permanent, regardless of any system failure. 
    - Ex: A user updates a record as part of a transaction. The database successfully updates the record. Right after this update, a power failure occurs. The database maintains its state while there is no power. The power is resumed. The database serves the record as per last update when requested by the user.
    - This ACID property relates to the CAP theorem in a way that the database systems providing traditional ACID guarantees choose consistency over availability. So it ensures C+P.
    - Ex: User A attempts to update a record as part of a transaction. The database validates the value and the update is successfully applied. After the successful completion of the transaction, when Users B and C request the same record, the database provides the updated value to both the users.
- BASE
    - BASE (pun intended) is a database design principle leveraged by many distributed database systems.
    - It stands for Basically Available, Soft state and Eventual consistency.
    - When a database supports BASE, it favors availability over consistency. So, it ensures A+P.
    - BASE leverages optimistic concurrency by relaxing the strong consistency constraints mandated by the ACID properties.
    - Basically available means that the database will always acknowledge a client’s request. 
    - This database is basically available, even though it has been partitioned as a result of a network failure. It can just return a failure response for the user request in this case. 
    - Soft state means that a database may be in an inconsistent state when data is read.
    - The results may change if the same data is requested again. 
    - Ex: User A updates a record on Peer A. Before the other peers are updated, User B requests the same record from Peer C. The database is now in a soft state, and stale data is returned to User B.
    - Eventual consistency means that the database only attains consistency once the changes have been propagated to all nodes.
    - Ex: User A updates a record. The record only gets updated at Peer A, but before the other peers can be updated, User B requests the same record. The database is now in a soft state. Stale data is returned to User B from Peer C. However, the consistency is eventually attained, and User C gets the correct value.
- ACID vs BASE
    - ACID ensures immediate consistency at the expense of availability due to the record locking.
    - BASE emphasizes availability over immediate consistency.
    - This soft approach toward consistency allows BASE-compliant databases to serve multiple clients without any latency though serving inconsistent results.
    - However, BASE-compliant databases are not useful for transactional systems where lack of consistency is a concern and needed.
    - A distributed database system may choose to provide some ACID properties.



