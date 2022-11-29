# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## 机器学习系统和湖仓一体化
# MAGIC ### (2022/11/20)
# MAGIC 
# MAGIC #### Dataware warehouse和Data lake的差异？ 
# MAGIC - Data warehouse-数仓不能处理非结构化数据，一定要有schema，不够flexible,存储成本高。主要定位场景为BI analytics。 Data lake-数据湖可以存储非结构化数据，不一定要有schema，灵活且存储成本低。主要定位场景为machine learning/deep learning。缺少schema，数据质量相对比数仓差一些
# MAGIC 
# MAGIC #### Databricks如何通过湖仓一体化，来实现在数据湖的基础上构建数据仓库？
# MAGIC - 因为Deltalake的数据format来实现，在parquet基础上增加了transaction log（versioning log, travel back, roll back）的layer，这个layer可以帮助在parquet基础上来增加数据管理和数据治理的功能，进而提升数据品质和可靠性。通知也增加了数据处理和查询速度，增加了很多indexing的功能
# MAGIC 
# MAGIC •	对于DS Modeling会很有帮助，建模过程中很重要的一环就是feature enginering，建模过程中可能会有不同版本的featurs，不同版本的model可能需要在不同的feature下train一下，看哪一套feature的性能最好。基于experiment/job的log功能，可以直接看出哪一个版本的training和performance tuning的结果最好
# MAGIC 
# MAGIC •	Subscriptions - 服务主体/catalogs
# MAGIC 
# MAGIC 
# MAGIC #### ML系统为什么复杂以及具有独特性？ 
# MAGIC - ①它是data centric模式系统，模型性能很大程度上取决于数据质量和数据量，数据总是在在变化。数据在变化，ML模型需要根据data的变化而变化，重新训练，要做模型监控，模型版本控制 
# MAGIC - ②要做数据预处理，清洗，来提高数据质量，数据cleaning和transformation 
# MAGIC - ③需要特征工程，"data is not ML usable"，需要工程化的特征处理之后才能使用 
# MAGIC - ④数据不一定及时，有延迟
# MAGIC 
# MAGIC #### DS技能要求：
# MAGIC - 1-data engineering;知道如何用python，spark，scala去写large-scale的特征工程的pipeline，去调取大数据 
# MAGIC - 2-模型会用python实现，ML的library有哪些及如何调用，mathematic有很深入的了解，对model performance tuning有很好的技术，才能做好model training的工作 
# MAGIC - 3-模型部署，涉及到底层IT的infrastructure，涉及到跟其他的系统如何对接。ML model不能直接给business user来使用，需要把它嵌入到一个系统中去。可能需要做一个REST Endpoint(API,REST是一组软件架构规范，并非协议或标准。API 开发人员可以采用各种方式实施 REST-即“表现层状态转移”（Representational State Transfer)，REST指的是一组架构约束条件和原则)，可能需要package model到container image，可能需要对Kubernetes有一定的了解（Deploy时有Azure Kubernetes Service的选项，K8S是一种可自动实施Linux 容器操作的开源平台）
# MAGIC 
# MAGIC •	ML是非常dynamic的ecosystem。首先算法相关的library差异很大，又有不同的技术流派：scikit-learn，tensorflow+keras，pytorch，huggingface。ML lifecycle技术，如ML Flow。对应的application也差异万千，如CV,NLP,Reinforcement Learning，Semi-supervised learning。ML system处于不断变化的一个状态，对Monitor的要求非常的高，不仅要监控data，也要监控model performance。需要考虑在什么条件下，要再次trigger一个模型re-train的行为，要把整个过程自动化，是非常有挑战的事情，把data+AI放到同一个平台上是非常有必要的事情。
# MAGIC 
# MAGIC 
# MAGIC #### 如何搭建一个end-2-end的ML system？ 
# MAGIC - 1-data & feature engineering pipeline(data extaction, data processing, data quality checking, feature engineering) 
# MAGIC Steps: data ingestion - ETL framework - workflow orchestration
# MAGIC 
# MAGIC - 2-feature store(将所有的feature存储到feature store，一方面用于model training，一方面用于model serving)
# MAGIC source raw data - feature - ML model
# MAGIC 
# MAGIC - 3-ML model training/retraining pipeline (model experimentation, model evaulation, metrics|parameters and model logging, model registry) 
# MAGIC (ML Runtime). ML training environment (jupyter + open source library etc.) - distributed ML training (spark)  - GPU cluster - <single node + CPU parallelization | single GPU node + multiple GPU core | mutiple node + mutiple core> - Experiment logs(MLflow tracking server = parameters + metrics + artifacts + metadata + models) - Model Registry(centralized的模型生命周期Lifecycle的管理中心) <Model Train(dev,data scientist role) - Model testing/validation(stg-staging,machine learning engineer role) - Model deploy(prd,deployment engineer role)> infrastructure engineer/devops engineer.
# MAGIC testing = unit testing + integration testing.
# MAGIC 
# MAGIC - 4-model metadata
# MAGIC 不同形式的model deployment. 1-contianer 2-large-scale batch processing 3-low latency online real time serving (RESTAPI) 4-edge deployment
# MAGIC 
# MAGIC - 5-model servering pipeline (model prediction, model validation, model packaging, model serving) 
# MAGIC model deployment - 不同形式不同应用，batch scoring - use model inference (batch/real time) - deploy in notebook workflow - model orchestration | online serving - enable serverless endpoint (Databricks, QPS - query per second 3000), 自动生成restAPI
# MAGIC 
# MAGIC - 6-ML model in production: model monitoring 
# MAGIC model production monitoring. data drifting monitor - scoring data | raw data, feature input data monitor, model output data. (classification - proportion of each classes | model accuracy | model metrics | model improvement | model A/B testing) stable or improvement? 
# MAGIC statistics values | training vs scoring | scoring vs scoring | model AUC, confusion matrix etc. | DLT(ETL framework) - Delta live table can be customized | in-platform solution | business driven customized metrics 
# MAGIC 
# MAGIC 
# MAGIC - 7-machine learning pipelines 
# MAGIC ML flow 2.0 pipelines(pipeline .yaml files)
# MAGIC 
# MAGIC - 8-workflow orchestration 
# MAGIC rule = "orchestrate any task from any platform"
# MAGIC eg: delta live table(DLT) task, auto loader task, Mlflow task, Python task, Java task, Spark task, SQL task, DBT task etc.	
# MAGIC workflow process example: data ingestion - data filtering/quality assurance/transformation - ML feature extraction - Persisting featurs/training prediction model
# MAGIC 
# MAGIC - 9-continuous integration, continuous training, continuous deployment(CI,CT,CD)+IaC-Infrastructure as code.
# MAGIC MLOps - 涉及到很多底层的infrastructure，80%的资源和时间都花在了devops上。MLOps stacks - 简单理解为很通用的CI/CD的infrastructure templates
# MAGIC Default MLOps Stacks： 1-CI/CD：github actions，azure devops 2-Infra-as-code：Terraform(framework + template) 3-Orchestrator：databricks workflow 4-Infra components: jobs/experiments/models