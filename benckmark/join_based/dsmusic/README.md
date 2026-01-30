## Overview

**Discogs and Spotify Music（short for DSMusic）** 是一个面向 Data Lake(House) 场景下弱关联（Join关系）的表格数据集。该数据集聚焦于音乐场景，包含两张弱关联的表格，分别为任务表（Discogs数据库）和辅助表（Spotify曲目）。任务表（Discogs_music.csv）包含Discogs公开存档的音乐信息，共计11000条数据；辅助表（Spotify_music.csv）包含存在于Spotify音乐平台的音乐信息，共计11000条数据。本数据集的任务为预测音乐类别（单标签多分类任务）。  

两张表之间存在弱关联（Join）关系，可以利用辅助表中的信息提升任务表的机器学习效果，以实现更精准的关联分析。

## Data Processing

DSMuscic数据集包含两张弱关联的数据表（任务表与辅助表），数据来源为Discogs数据库和Spotify在线音乐平台。以下为数据处理步骤的详细说明：

**任务表：** Discogs数据库中的音乐（Discogs_music.csv）  
- **数据来源：**[Discogs_music](https://www.kaggle.com/datasets/fleshmetal/records-a-comprehensive-music-metadata-dataset)  
- **数据清洗：**
删去表中styles列（可能对预测音乐genres造成干扰）。
- **数据筛选：**
根据genres列，提取出其中的单标签类别，从得到的每个单标签类别中随机抽取1000个样本作为任务表数据，共11个类别。

**辅助表：** Spotify音乐数据集（Spotify_music.csv）    
- **数据来源：**[Spotify_music](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)  
- **数据筛选：**
根据任务表进行1NN匹配，保留任务表匹配到的样本。

## Dataset Composition

- **Discogs_music.csv** :该文件包含11000条Discogs数据库中的音乐曲目信息，清洗后具有 5 个特征，共有 11 种音乐类别。 

- **Spotify_music.csv** :该文件包含11000条音乐曲目信息，具有20个特征。 

- **map.csv** :该文件包含我们根据音乐名采取1NN匹配得到的映射关系，可利用其进行join操作，从而利用Spotify_music.csv提供特征增量，共三列，结构为：(T1_index, T2_index, cosine_similarity)。 
<p>

- **mask.pt** :该文件为任务表Discogs_music.csv的train/val/test 划分，因数据集各类别样本平衡，我们将各类别样本随机以70%/10%/20%的比例划分入train/val/test。mask.pt 文件存储了一个字典，其中包含 train_mask、val_mask 和 test_mask 三个键，每个键对应一个与样本总数(11000)相同长度的布尔张量，分别用于标识样本属于训练集、验证集或测试集。

## References
