---
layout: post
title: Apache Flink
date: 2025-12-22 19:54:00-0400
featured: false
description: Realtime and Big Data Analytics Course at NYU Courant - Personal Notes 10
tags: 
categories: RBDA-NYU
giscus_comments: true
related_posts: false

---

**Slide 17**

Stateful stream processing refers to the design of applications that handle continuous, unbounded streams of events (like user clicks, orders, server logs, or sensor readings) while maintaining state—intermediate data that summarizes, aggregates, or tracks information across multiple events. Unlike stateless processing, where each event is handled independently, stateful processing allows the application to remember past events and use that information to compute results for new events.

For example, a stateful stream processing application could:
- Track the running count of clicks per user on a website.
- Maintain a windowed sum of sales per product over the last hour.
- Store sensor readings to compute rolling averages or detect anomalies.

In practice, the application reacts to each incoming event by reading from and/or writing to its state, performing computations that can depend on both the new event and historical information stored in the state. This enables complex operations like aggregations, joins, or pattern detection on real-time streams of data.

State here is durable and fault-tolerant (managed by the stream processing framework, e.g., Flink), so even if the application crashes or is restarted, the state can be recovered and processing can continue correctly.

**Slide 19 & 20**

Stateful stream processing applications often consume events from an event log such as Kafka. An event log is a durable, append-only store of events, which guarantees that events are written in order and that this order cannot be changed. This design allows multiple consumers to read the same stream independently and ensures that all consumers see events in the exact same order.

When a stateful application like Flink connects to an event log, the log provides durable storage and deterministic replay of events. Each incoming event can be processed while updating the application’s state. If the application crashes or fails, Flink can restore the last consistent snapshot of its state from a checkpoint and reset its read position on the event log. It then replays events from that point onward until it catches up to the latest event in the stream.

This mechanism ensures fault tolerance and exactly-once semantics: the application can recover from failures without losing data or processing events out of order. It also allows the same input stream to be reprocessed multiple times if needed—for example, to recompute results after a logic update—because the event log persists all events in order.

In essence, the combination of stateful processing in Flink and the durable, ordered nature of event logs allows applications to maintain correct state, recover from failures, and guarantee deterministic processing over unbounded streams.

**Slide 69-72**

In stream processing, “one minute” can mean different things depending on the time semantics you choose, and this choice has a big impact on correctness and predictability.

With processing time, “one minute” refers to one minute on the wall clock of the machine running the operator. A processing-time window simply groups together all events that arrive at the operator within that one-minute interval, regardless of when those events actually occurred in the real world. This makes processing-time windows fast and simple, but also sensitive to system behavior: network delays, backpressure, retries, or slower machines can shift events into different windows, meaning the same input stream may produce different results at different times or under different loads.

With event time, “one minute” refers to one minute in the timeline of the events themselves, based on timestamps attached to each event (for example, when a user clicked a button or a sensor recorded a measurement). An event-time window groups events by when they happened, not when they arrived. Because the timestamps are part of the data, event time decouples processing speed from results: whether the stream is processed fast or slow, or events arrive late or out of order, the window computation is based on the same logical time boundaries. As a result, event-time operations are predictable and deterministic, yielding the same output as long as the input events and their timestamps are the same.

**Slide 73-76**

Watermarks are how a stream processor like Flink decides when an event-time window is complete and ready to be evaluated. Since events can arrive late or out of order, the system cannot rely on wall-clock time. Instead, a watermark acts as a logical event-time clock that advances as the system gains confidence about how far it has progressed in event time.

Conceptually, a watermark with timestamp T means: “I believe that all events with event-time ≤ T have already arrived.” When an operator receives this watermark, it can safely close event-time windows that end at or before T, trigger their computation, and emit results. This makes watermarks essential for event-time windows and for correctly handling out-of-order events, because they tell operators when to stop waiting for older timestamps.

Watermarks introduce a tradeoff between latency and correctness (confidence). If watermarks advance aggressively (eager watermarks), results are produced quickly, but there is a higher risk that late events will show up after a window has already been computed. If watermarks advance conservatively (relaxed watermarks), the system waits longer before triggering windows, increasing confidence that all relevant events have arrived—but at the cost of higher end-to-end latency.

Because late events can still occur even after a watermark, stream processing systems must define how to handle them. Common strategies include dropping late events, logging or redirecting them for monitoring, or using them to update or correct previously emitted results (for example, via retractions or updates). This is why watermarks alone are not enough—the system’s late-event handling policy is just as important for correctness.

In Flink, watermarks are defined by the developer and generated by the source operator of the stream. When a Flink job reads events from a source such as Kafka, the developer configures a watermark strategy that tells Flink how to extract event-time timestamps from each event and how much out-of-orderness (lateness) is allowed. Based on this logic, the source periodically emits watermarks that signal the system’s progress in event time, indicating that no more events with timestamps earlier than a certain time are expected. These watermarks then flow through the entire dataflow alongside the events, and downstream operators such as windows and joins use them to decide when it is safe to trigger computations. Operators themselves do not create watermarks; they only act upon the watermarks produced by the source.

**Slide 77**

Even though event time gives correct and deterministic results, processing time is still useful because it optimizes for speed and simplicity rather than correctness under disorder. Processing-time windows trigger based on the machine’s wall clock, so results are produced immediately as data arrives, giving the lowest possible latency with no need to wait for late events or watermarks. This makes them ideal for real-time monitoring, dashboards, and alerting, where users care more about seeing what is happening now than about perfectly accurate historical results. In addition, processing time reflects the actual arrival pattern of the stream, including delays and bursts, which can be valuable when analyzing system behavior or load rather than the underlying business events.

**Slide 116-119**

In Flink, state is always tied to a specific operator, and operators must register their state so that the runtime can track and manage it. There are two main types of state: operator state and keyed state. Operator state is scoped to a single operator task; all records processed by that task share the same state, but it cannot be accessed by other tasks of the same or different operators. Keyed state, on the other hand, is partitioned by a key defined in the records of the input stream. Each key has its own state instance, and all records with the same key are routed to the operator task that maintains that key’s state.

To enable efficient state access with low latency, Flink keeps the state locally within each parallel task. How the state is stored, accessed, and maintained is handled by a state backend, such as RocksDB. The state backend is responsible for managing the local state for fast access during processing and for checkpointing the state to a remote location to enable fault tolerance and recovery. This design allows Flink to maintain consistent, high-performance state even in large-scale, distributed streaming applications.

**Slide 121**

Flink’s recovery mechanism relies on consistent checkpoints of the application’s state. A checkpoint is a snapshot of the state of all tasks in the streaming application at a specific point in time. For the checkpoint to be consistent, it must capture the state such that all tasks have processed exactly the same input events up to that point. This ensures that, in case of a failure, the application can be restored to a previous checkpoint and resume processing without losing data or processing events out of order.

**Slide 123**

Flink can provide exactly-once state consistency through its checkpointing and recovery mechanism, but only if certain conditions are met. First, all operators must correctly checkpoint and restore their state, so that after a failure, the internal state of the application reflects a precise moment in time. Second, all input streams must be reset to the exact position they were at when the checkpoint was taken, so the same events can be replayed deterministically. Whether this is possible depends on the data source: event logs like Kafka support resetting to a previous offset, allowing Flink to reprocess events reliably, whereas sources like sockets cannot be reset, since consumed data is lost. When both state restoration and input replay are possible, Flink can guarantee that each event affects the application state exactly once, even in the presence of failures.

**Slide 124**

Flink’s checkpointing and recovery mechanism guarantees exactly-once consistency only for the application’s internal state, not automatically for the external systems where results are written. When a failure occurs, Flink restores operator state from the latest checkpoint and replays input events, which can cause some output records to be emitted again. If the sink does not support transactions or idempotent writes (for example, a plain filesystem, database, or event log), these repeated emissions may lead to duplicate results in downstream systems. To achieve end-to-end exactly-once semantics, the sink itself must be able to handle duplicates safely or participate in Flink’s transactional mechanisms.

**Slide 125**

End-to-end exactly-once consistency is achieved when both Flink’s internal state and the external sink observe each event exactly once. Flink provides special exactly-once sink implementations for some storage systems (such as Kafka or transactional filesystems) that buffer output and only commit the emitted records when a checkpoint successfully completes, ensuring that partial results from failed executions are never made visible. For storage systems that do not support transactions, idempotent updates are commonly used: the sink is designed so that writing the same record multiple times produces the same final result (for example, using upserts or overwriting by a unique key). With transactional sinks or idempotent writes, replayed events during recovery do not cause incorrect or duplicated outcomes, enabling true end-to-end exactly-once behavior.

**Slide 126 - 136**

Flink’s checkpointing algorithm is based on the Chandy–Lamport distributed snapshot algorithm, which allows the system to take a consistent snapshot of a running distributed application without stopping the entire dataflow. Instead of pausing all processing, Flink decouples checkpointing from normal execution so that tasks can keep processing data while checkpoints are being coordinated and state is being persisted. This is crucial for low-latency stream processing.

The core mechanism behind this is the checkpoint barrier. A checkpoint barrier is a special marker record that Flink’s source operators inject into the data stream when a checkpoint is triggered by the JobManager. Each barrier carries a checkpoint ID and flows through the same channels as normal records, but it cannot be overtaken by other records. Conceptually, the barrier splits the stream: all state updates caused by records before the barrier belong to the current checkpoint, and updates caused by records after the barrier belong to a future checkpoint.

When a checkpoint starts, the JobManager instructs all source tasks to begin a new checkpoint. Each source temporarily pauses emitting records, snapshots its local state to the configured state backend, and then broadcasts checkpoint barriers on all its outgoing streams. Once the state snapshot is complete, the source acknowledges the checkpoint to the JobManager and resumes normal data emission.

When an intermediate or downstream task receives a checkpoint barrier, it must ensure consistency across all its inputs. The task performs barrier alignment: it waits until it has received the same checkpoint barrier from all its input partitions. While waiting, it continues processing records from inputs that have not yet sent a barrier, but buffers records from inputs that already forwarded a barrier. This guarantees that no record belonging to the “future” checkpoint is processed too early.

As soon as the task has received barriers from all inputs, it snapshots its own state to the state backend, forwards the checkpoint barrier to its downstream tasks, and then resumes processing by draining the buffered records. This process continues through the entire dataflow graph. Finally, when the JobManager has received checkpoint acknowledgements from all tasks, it marks the checkpoint as complete. At that point, Flink has a globally consistent snapshot of the application state, which can be used for recovery and provides the foundation for exactly-once state consistency.
