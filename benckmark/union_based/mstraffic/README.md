## Overview

**Maryland and Seattle Traffic Collisions (MSTraffic)** is a tabular dataset designed for weakly related (Union-based) table scenarios in Data Lake(House) settings. The dataset focuses on traffic collision incidents and comprises two weakly related tables: a task table (Maryland collision reports) and an auxiliary table (Seattle collision reports). The task table (Maryland.csv) contains 10,800 traffic collision records from January 2017 to December 2023, while the auxiliary table (Seattle.csv) contains 10,800 traffic collision records from January 2014 to December 2023. The objective of this dataset is to predict the collision type of traffic incidents (a classification task).

The two tables exhibit a weak association (Union relationship), where information from the auxiliary table can be leveraged to enhance machine learning performance on the task table, thereby enabling more accurate relational analysis.

## Data Processing

The MSTraffic dataset consists of two weakly related traffic collision tables (a task table and an auxiliary table), derived from publicly available traffic collision reports. The detailed data processing steps are described below:

**Task Table:** Maryland Collision Reports (Maryland.csv)
- **Data Source:** [Crash Reporting - Incidents Data](https://catalog.data.gov/dataset/crash-reporting-incidents-data)
- **Data Cleaning:**
Samples with Collision Type (label) values of OTHER, UNKNOWN, Other, Unknown, or null were removed.
- **Data Filtering:**
Records from 2017 to 2023 were selected based on the Crash Date/Time column. Categories with more than 1,200 samples were retained, and 1,200 samples were randomly drawn from each category to form the task table.

**Auxiliary Table:** Seattle Collision Reports (Seattle.csv)
- **Data Source:** [SDOT Collisions All Years](https://catalog.data.gov/dataset/sdot-collisions-all-years-2a008)
- **Data Cleaning:**
Samples with COLLISIONTYPE (label) values of "other" or null were removed.
- **Data Filtering:**
Records from 2014 to 2023 were selected based on the INCDATE column. Categories with more than 1,200 samples were retained, and 1,200 samples were randomly drawn from each category to form the auxiliary table.

## Dataset Composition

- **Maryland.csv**: This file contains 10,800 collision records, each described by 37 features including collision type, report number, occurrence time, lane direction, route type, weather conditions, and road surface conditions. The collision type label comprises 9 categories:
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

- **Seattle.csv**: This file contains 10,800 collision records, each described by 50 features including collision type, report number, occurrence time, incident location, severity, and whether speeding was involved. The collision type label comprises 9 categories:
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

- **mask.pt**: This file provides the train/validation/test split for the task table (Maryland.csv). Since the dataset is class-balanced, samples within each category are sorted chronologically (from oldest to newest) and partitioned into train/validation/test sets at a ratio of 70%/10%/20%, respectively. The mask.pt file stores a dictionary containing three keys — train_mask, val_mask, and test_mask — each corresponding to a Boolean tensor of the same length as the total number of samples (10,800), indicating whether each sample belongs to the training, validation, or test set.

- **mapping.csv**: This file contains row-level matching results between the task table (Maryland.csv) and the auxiliary table (Seattle.csv) for feature augmentation. Given the absence of explicit join keys, we employ row-text 1-NN matching: each row is represented as a textual sequence (concatenated column name–cell value pairs), encoded into embeddings, and matched via 1-NN in the embedding space. The file comprises four columns: T1_index (task table row index), T2_index (matched auxiliary table row index), cosine_similarity, and cosine_distance. This mapping enables feature augmentation by concatenating matched auxiliary features to target records.

## References
