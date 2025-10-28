---
layout: post
title: Introduction to Realtime and Big Data Analytics
date: 2025-10-22 11:23:00-0400
featured: false
description: Notes on Big Data (the 5 V's & data types) and the Four Categories of Data Analytics (Descriptive, Diagnostic, Predictive, and Prescriptive).
tags: RBDA
categories: RBDA-NYU
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

- Big Data is a field dedicated to the analysis, processing, and storage of large collections of data that frequently originate from disparate sources.
- Big Data addresses distinct requirements: I) Combine multiple unrelated datasets. II) Process large amounts of unstructured data. III) Harvest hidden information in a time-sensitive manner.
- A dataset is a collection or group of related data.
- Each dataset member (“datum”) shares the same set of attributes or properties as others in the same dataset.
- Ex: An extract of rows from a database table stored in a CSV formatted file.
- Data analysis is the process of examining data to find facts, relationships, patterns, insights and/or trends.
- The overall goal of data analysis is to support better decision-making.
- Data analytics is a discipline that includes the management of the complete data lifecycle, which encompasses collecting, cleaning, organizing, storing, analyzing and governing data.
- Ex: In business-oriented environments, data analytics results can lower operational costs and facilitate strategic decision-making. In the scientific domain, data analytics can help identify the cause of a phenomenon to improve the accuracy of predictions. In service-based environments, data analytics can help strengthen the focus on delivering high-quality services by driving down costs.
- There are four general categories of analytics that are distinguished by the results they produce: descriptive, diagnostic, predictive and prescriptive analysis.
- Descriptive analytics aim to answer questions about events that have already occurred. Descriptive analytics contextualizes data to generate information.
- The operational systems (e.g., OLTP, CRM, ERP) are queried via descriptive analytics tools to generate static reports or dashboards.
- Diagnostic analysts aim to determine the cause of a phenomenon that occurred in the past using questions that focus on the reason behind the event.
- The goal is to determine what information is related to the phenomenon in order to answer questions that seek to determine why something has occurred. 
- Diagnostic analytics usually collect data from multiple sources and store it in a structure (e.g., OLAP) so that users can perform interactive drill-down and roll-up analysis.
- Predictive analytics aim to determine the outcome of an event that might occur in the future.
- Information is associated to build models that are used to generate future predictions based upon past events. 
- Predictive analytics use large datasets of internal and external data and various data analysis techniques to provide user-friendly front-end interfaces.
- Prescriptive analytics build upon the results of predictive analytics by prescribing actions that should be taken. 
- The focus is not only on what prescribed option is best to follow, but why.
- Prescriptive analytics use business rules and large amounts of internal and external data to simulate outcomes and prescribe the best course of action.

<div class="row justify-content-center">
    <div class="col-sm-9 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/bd-1.png" title="bd-1" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

- Big data characteristics, the five V’s: Volume, Velocity, Variety, Veracity and Value.
- Veracity refers to the quality of data. Data with a high signal-to-noise ratio has more veracity. 
- Value is defined as the usefulness of data for an enterprise.
- Value is also impacted by data lifecycle-related concerns, like how well has the data been stored? Were valuable attributes of data removed during data cleansing? Are the right types of questions being asked during data analysis? Are the results of the analysis being accurately communicated to the appropriate decision-makers?
- There are different types of data. Based on source, we’ve Human-generated data and Machine-generated data. Based on format, we’ve structured, unstructured, semi-structured data and metadata.
- Structured data conforms to a data model or schema. It is used to capture relationships between different entities and is therefore most often stored in relational database. Ex: banking transactions, invoices, customer records, etc.
- Unstructured data doesn’t conform to a data model or schema. It is either textual or binary and often conveyed via files that are self-contained and non-relational. The majority of data is unstructured. Ex: textual data, video, image files, audio, etc.
- Semi-structured data has a defined level of structure and consistency, but is not relational in nature. Instead, it is hierarchical or graph based. This kind of data is commonly stored in text-based files. Ex: XML data, JSON data, sensor data(CSV), etc.
- Metadata provides information about a dataset’s characteristics and structure. It is mostly machine-generated and can be appended to data. Ex: XML tags providing the author and creation date of a document, Attributes providing the file size and resolution of a digital photograph, etc.
- Big data solutions need to support multiple formats and types of data.
