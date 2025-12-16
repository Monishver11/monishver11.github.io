---
layout: post
title: MapReduce Design Patterns
date: 2025-12-15 18:40:00-0400
featured: false
description: Realtime and Big Data Analytics Course at NYU Courant - Conceptual Notes 5
tags: 
categories: RBDA-NYU
giscus_comments: true
related_posts: false

---

#### **Serialization**

- Serialization is the process of turning structured objects into a byte stream for transmission over a network or for writing to persistent storage.
- Deserialization is the reverse process of turning a byte stream back into a series of structured objects.
- A good serialization format should be compact, fast, extensible and interoperable.
- Hadoop uses its own serialization format: Writable.
- Hadoop comes with a large selection of Writable classes, which are available in the org.apache.hadoop.io package.
- There are Writable wrappers for all the Java primitive types except char (which can be stored in an IntWritable).
- All Writable wrappers have a get() and set() method for retrieving and storing the wrapped value. Example: IntWritable count = new IntWritable(42).

<div class="row justify-content-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/mr-dp1.png" title="mr-dp-1" class="img-fluid rounded z-depth-1" %}
    </div>
</div>￼

- Text is the Writable wrapper for mutable UTF-8 strings. Ex: Text word = new Text("Hadoop");
- BytesWritable is the Writable wrapper for byte[]. 
- NullWritable is a special type of Writable, which has zero-length serialization. No bytes are written to or read from the stream. It is used as a placeholder.
- For example, in MapReduce, a key or a value can be declared as a NullWritable when you don’t need to use that position, eﬀectively storing a constant empty value. It is an immutable singleton, and the instance can be retrieved by calling NullWritable.get(). Ex: NullWritable nullKey = NullWritable.get();

#### **Counters**
- Counters are a useful channel for gathering statistics about the job: For quality control (Example: what’s the percentage of records that are invalid?), For application-level statistics (Example: how many users in the dataset are between the ages of 18—64?)
- MapReduce allows user code to define a set of counters, which are then incremented as desired in the mapper or reducer.
- Counters are defined by a Java enum, which serves to group related counters.
- A job may define any number of enums, each with any number of fields. The name of the enum is the group name. The enum’s fields are the counter names.
- Counters are global: the MapReduce framework aggregates them across all mappers and reducers to produce a grand total at the end of the job.
- Ex: enum Temperature { MISSING, MALFORMED}
- context.getCounter(Temperature.MALFORMED).increment(1);
- context.getCounter(Temperature.MISSING).increment(1);
- context.getCounter("TemperatureQuality", parser.getQuality()).increment(1); //dynamic counter, here "TemperatureQuality" is a manual group name (not an enum).
- Note: In Hadoop, counters are defined and incremented by the Mapper or Reducer, tracked locally by the NodeManager, aggregated by the Application Master, and finally reported to the client by the Resource Manager at job completion.
- Note: Hadoop also provides built-in counter groups such as FileSystemCounters (bytes read/written), TaskCounters (records processed, spilled data), and JobCounters (launched or failed tasks).
- Ex: Counter hdfsRead = context.getCounter("FileSystemCounters", "HDFS_BYTES_READ"); //access built-in FileSystem counter

#### **MapReduce design patterns**
- Summarization patterns
  - Numerical summarizations
  - Inverted index summarizations
  - Counting with counters
- Filtering patterns
  - Filtering
  - Bloom filtering
  - Top ten
  - Distinct
- Data organization patterns
  - Structured to hierarchical
  - Partitioning
  - Binning
  - Total order sorting
  - Shuﬄing
- Join patterns
  - Reduce-side join
  - Replicated join
  - Cartesian product
- Metapatterns
  - Job chaining
  - Chain folding
  - Job merging
- Input and output patterns

**TODO:** Add details, explanation and images to MapReduce design patterns;