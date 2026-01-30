## Overview

**German and Australian Used Cars (short for GACars)** 是一个面向 Data Lake(House) 场景下弱关联（Union关系）的表格数据集。该数据集聚焦于二手车交易场景。包含两张弱关联的表格，分别为任务表（German二手车售卖信息）和辅助表（Australian二手车售卖信息）。任务表（german.csv）为2023年抓取的二手车售卖信息，共计13,000条数据；辅助表（australian.csv）为2023年抓取的二手车售卖信息，共计3,600条数据。本数据集的任务为预测二手车售卖的价格区间（分类任务）。

两张表之间存在弱关联（Union）关系，可以利用辅助表中的信息提升任务表的机器学习效果，以实现更精准的关联分析。

## Data Processing

GACars 数据集包含两张弱关联的二手车售卖信息表（任务表与辅助表），数据来源为Kaggle经典的数据集。以下为数据处理步骤的详细说明：

**任务表：** German二手车信息（german.csv）  
- **数据来源：**[Germany Used Cars Dataset 2023](https://www.kaggle.com/datasets/wspirat/germany-used-cars-dataset-2023)  
- **数据处理：**
对二手车挂牌价格划分为10个区间，除最后一个区间(more than 90000 EUR)外，区间长度为10,000 EUR。对10个区间随机提取各1300个样本作为任务表数据。

**辅助表：** Australian二手车信息（australian.csv）  
- **数据来源：**[Australian Vehicle Prices](https://www.kaggle.com/datasets/nelgiriyewithana/australian-vehicle-prices) 
- **数据处理：**
对二手车挂牌价格划分为8个区间，除最后一个区间(more than 70000 AUD)外，区间长度为10,000 AUD。对8个区间随机提取各450个样本作为任务表数据。

## Dataset Composition

- **german.csv** :该文件包含13,000条二手车售卖信息，具有 15 个字段，共有10个价格区间类别。 

- **australian.csv** :该文件包含3,600条二手车售卖信息，具有 19 个字段，共有8个价格区间类别。 

- **mask.pt** :该文件为任务表german.csv的train/val/test 划分，因数据集各类别样本平衡，我们随机将各类别样本以70%/10%/20%的比例划分入train/val/test。mask.pt 文件存储了一个字典，其中包含 train_mask、val_mask 和 test_mask 三个键，每个键对应一个与样本总数(13,000)相同长度的布尔张量，分别用于标识样本属于训练集、验证集或测试集。

## References
