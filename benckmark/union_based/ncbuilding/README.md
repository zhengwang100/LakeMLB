## Overview

**NewYork and Chicago Buliding Violations (short for NCBuilding)** 是一个面向 Data Lake(House) 场景下弱关联（Union关系）的表格数据集。该数据集聚焦于违规建筑投诉场景，包含两张弱关联的表格，分别为任务表（New York投诉报告）和辅助表（Chicago投诉报告）。任务表（newyork.csv）包含从2023年1月至2024年12月期间的违规建筑投诉记录，共计30,000条数据；辅助表（chicago.csv）包含2023年1月至2024年12月期间的违规建筑投诉记录，为37,000条数据。本数据集的任务为预测所投诉的建筑违规类型（分类任务）。  

两张表之间存在弱关联（Union）关系，可以利用辅助表中的信息提升任务表的机器学习效果，以实现更精准的关联分析。

## Data Processing

NCBuilding 数据集包含两张弱关联的违规建筑投诉数据表（任务表与辅助表），数据来源为公开的违规建筑投诉报告。以下为数据处理步骤的详细说明：

**任务表：** New York投诉报告（newyork.csv）  
- **数据来源：**[Housing Maintenance Code Violations](https://catalog.data.gov/dataset/housing-maintenance-code-violations)  
- **数据清洗与筛选：**
根据 InspectionDate 列，选取日期2023年1月至2024年12月期间的记录。从NOVDescription中提取每条记录具体的违规条目，保存为StatuteCodes列，随后删除多标签的样本。为防止标签泄露，删除了OrderNumber, NOVDescription列，因为这两列与标签列近乎呈一一对应关系。提取StatuteCodes样本数大于1000的类别，各类提取1000个样本，共计30,000条记录作为任务表数据。


**辅助表：** Chicago投诉报告（chicago.csv）  
- **数据来源：**[Building Violations](https://catalog.data.gov/dataset/building-violations)  
- **数据清洗与筛选：**
为防止标签泄露，事先删除了VIOLATION CODE, DEPARTMENT BUREAU, VIOLATION ORDINANCE列，因为这两列与标签列近乎呈一一对应关系。根据 VIOLATION LAST MODIFIED DATE 列，选取日期2023年1月至2024年12月期间的记录，提取VIOLATION DESCRIPTION列样本数大于1000的类别，各类提取1000个样本，共计37,000条记录作为任务表数据。

## Dataset Composition
- **newyork.csv** :该文件包含30,000条建筑投诉记录，具有 40 个字段，共有30个建筑违规类别。  

- **chicago.csv** :该文件包含37,000条建筑投诉记录，具有 23 个字段，共有37个建筑违规类别。

- **mask.pt** :该文件为任务表newyork.csv的train/val/test 划分，因数据集各类别样本平衡，我们按照时间顺序由旧到新将各类别样本依次以70%/10%/20%的比例划分入train/val/test。mask.pt 文件存储了一个字典，其中包含 train_mask、val_mask 和 test_mask 三个键，每个键对应一个与样本总数(30,000)相同长度的布尔张量，分别用于标识样本属于训练集、验证集或测试集。

## References
