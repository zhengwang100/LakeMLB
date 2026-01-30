## Overview

**Nasdaq and NYSE Stocks NNStocks (short for NNStocks)** 是一个面向 Data Lake(House) 场景下弱关联（Join关系）的表格数据集。该数据集聚焦于上市企业场景，包含两张弱关联的表格，分别为任务表（Nasdaq, NYSE上市公司）和辅助表（wiki百科）。任务表（stocks.csv）包含Nasdaq与NYSE的上市公司信息，共计1078条数据；辅助表（wiki.csv）包含对应存在于wiki百科的公司信息，共计937条数据。本数据集的任务为预测上市公司的行业类别（分类任务）。  

两张表之间存在弱关联（Join）关系，可以利用辅助表中的信息提升任务表的机器学习效果，以实现更精准的关联分析。

## Data Processing

NNStocks 数据集包含两张弱关联的数据表（任务表与辅助表），数据来源为Nasdaq与NYSE的官方名单与维基百科。以下为数据处理步骤的详细说明：

**任务表：** Nasdaq, NYSE上市公司（stocks.csv）  
- **数据来源：**[US-Stock-Symbols](https://github.com/rreichel3/US-Stock-Symbols)  
- **数据筛选：**
根据 sector 列，提取样本数大于98的类别（因保证类别样本均衡且足够，忽略了样本数为63的Miscellaneous类别），每类随机选取98个样本（如有配对的wiki信息则优先从中选取）作为任务表数据, 共11个类别。

**辅助表：** wiki百科信息（wiki.csv）  
- **数据清洗：**
将对应公司样本的主页infobox信息提取后保留空缺值在95%以内的属性。  
- **数据筛选：**
如果有实体匹配的wiki页面相同（例如主公司与子公司）则执行去重操作。

## Dataset Composition

- **stocks.csv** :该文件包含1078条上市公司信息，具有 29 个字段，共有 11 个行业类别。 

- **wiki.csv** :该文件包含937条能获取的wiki百科主页infobox信息，清洗后共有22个特征。 

- **map.csv** :该文件包含我们根据实体名采取1NN匹配得到的映射关系，可利用其进行join操作，从而利用wiki.csv提供特征增量，共三列结构为：(T1_index, T2_index, cosine_similarity)。 
<p>

- **mask.pt** :该文件为任务表stocks.csv的train/val/test 划分，因数据集各类别样本平衡，我们将各类别样本随机以70%/10%/20%的比例划分入train/val/test。mask.pt 文件存储了一个字典，其中包含 train_mask、val_mask 和 test_mask 三个键，每个键对应一个与样本总数(1078)相同长度的布尔张量，分别用于标识样本属于训练集、验证集或测试集。

## References
