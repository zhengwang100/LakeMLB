## Overview

**Maryland and Seattle Traffic Collisions (short for MSTraffic)** 是一个面向 Data Lake(House) 场景下弱关联（Union关系）的表格数据集。该数据集聚焦于交通事故场景，包含两张弱关联的表格，分别为任务表（Maryland事故报告）和辅助表（Seattle事故报告）。任务表（Maryland.csv）包含从2017年1月至2023年12月期间的交通事故记录，共计10,800条数据；辅助表（Seattle.csv）包含2014年1月至2023年12月期间的交通事故记录，同样为10,800条数据。本数据集的任务为预测交通事故的碰撞类型（分类任务）。  

两张表之间存在弱关联（Union）关系，可以利用辅助表中的信息提升任务表的机器学习效果，以实现更精准的关联分析。

## Data Processing

MSTraffic 数据集包含两张弱关联的交通事故数据表（任务表与辅助表），数据来源为公开的交通事故报告。以下为数据处理步骤的详细说明：

**任务表：** Maryland事故报告（Maryland.csv）  
- **数据来源：**[Crash Reporting - Incidents Data](https://catalog.data.gov/dataset/crash-reporting-incidents-data)  
- **数据清洗：**
删除 Collision Type（标签）列中类别为 OTHER、UNKNOWN、Other、Unknown 及空值的样本。  
- **数据筛选：**
根据 Crash Date/Time 列，选取2017~2023年的记录，并提取样本数大于1200的类别，每类随机选取1200个样本作为任务表数据。

**辅助表：** Seattle事故报告（Seattle.csv）  
- **数据来源：**[SDOT Collisions All Years](https://catalog.data.gov/dataset/sdot-collisions-all-years-2a008)  
- **数据清洗：**
删除 COLLISIONTYPE（标签）列中类别为 other 及空值的样本。  
- **数据筛选：**
根据 INCDATE 列，选取2014~2023年的记录，并提取样本数大于1200的类别，每类随机选取1200个样本作为辅助表数据。

## Dataset Composition

- **Maryland.csv** :该文件包含10,800条事故记录，每条事故具有碰撞类型、报告编号、发生时间、车道方向、路线类型、天气情况，路面状况等37个特征。其中碰撞类型有9个类别：  
  * SAME DIR REAR END
  * SINGLE VEHICLE
  * SAME DIRECTION SIDESWIPE
  * HEAD ON
  * STRAIGHT MOVEMENT ANGLE
  * HEAD ON LEFT TURN
  * OPPOSITE DIRECTION SIDESWIPE
  * SAME DIRECTION LEFT TURN
  * SAME DIRECTION RIGHT TURN
 
<p>

- **Seattle.csv** :该文件包含10,800条事故记录，每条事故具有碰撞类型、报告编号、发生时间、事故位置、严重程度，是否超速等50个特征。其中碰撞类型有9个类别：  
  * Rear Ended
  * Pedestrian
  * Sideswipe
  * Angles
  * Parked Car
  * Cycles
  * Left Turn
  * Head On
  * Right Turn

<p>

- **mask.pt** :该文件为任务表maryland.csv的train/val/test 划分，因数据集各类别样本平衡，我们按照时间顺序由旧到新将各类别样本依次以70%/10%/20%的比例划分入train/val/test。mask.pt 文件存储了一个字典，其中包含 train_mask、val_mask 和 test_mask 三个键，每个键对应一个与样本总数(10,800)相同长度的布尔张量，分别用于标识样本属于训练集、验证集或测试集。

## References
