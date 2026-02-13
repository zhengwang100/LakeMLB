## Overview

**German and Australian Used Cars (GACars)** is a tabular dataset designed for weakly related (Union-based) table scenarios in Data Lake(House) settings. The dataset focuses on used car sales transactions and comprises two weakly related tables: a task table (German used car listings) and an auxiliary table (Australian used car listings). The task table (german.csv) contains 13,000 used car listing records collected in 2023, while the auxiliary table (australian.csv) contains 3,600 used car listing records collected in 2023. The objective of this dataset is to predict the price range of used car sales (a classification task).

The two tables exhibit a weak association (Union relationship), where information from the auxiliary table can be leveraged to enhance machine learning performance on the task table, thereby enabling more accurate relational analysis.

## Data Processing

The GACars dataset consists of two weakly related used car listing tables (a task table and an auxiliary table), derived from classic Kaggle datasets. The detailed data processing steps are described below:

**Task Table:** German Used Car Listings (german.csv)
- **Data Source:** [Germany Used Cars Dataset 2023](https://www.kaggle.com/datasets/wspirat/germany-used-cars-dataset-2023)
- **Data Processing:**
The used car listing prices were divided into 10 intervals, with each interval spanning 10,000 EUR, except for the last interval (more than 90,000 EUR). A total of 1,300 samples were randomly extracted from each of the 10 intervals to form the task table.

**Auxiliary Table:** Australian Used Car Listings (australian.csv)
- **Data Source:** [Australian Vehicle Prices](https://www.kaggle.com/datasets/nelgiriyewithana/australian-vehicle-prices)
- **Data Processing:**
The used car listing prices were divided into 8 intervals, with each interval spanning 10,000 AUD, except for the last interval (more than 70,000 AUD). A total of 450 samples were randomly extracted from each of the 8 intervals to form the auxiliary table.

## Dataset Composition

- **german.csv**: This file contains 13,000 used car listing records with 15 features, covering 10 price range categories.

- **australian.csv**: This file contains 3,600 used car listing records with 19 features, covering 8 price range categories.

- **mask.pt**: This file provides the train/validation/test split for the task table (german.csv). Since the dataset is class-balanced, samples within each category are randomly partitioned into train/validation/test sets at a ratio of 70%/10%/20%, respectively. The mask.pt file stores a dictionary containing three keys — train_mask, val_mask, and test_mask — each corresponding to a Boolean tensor of the same length as the total number of samples (13,000), indicating whether each sample belongs to the training, validation, or test set.

- **mapping.csv**: This file contains row-level matching results between the task table (german.csv) and the auxiliary table (australian.csv) for feature augmentation. Given the absence of explicit join keys, we employ row-text 1-NN matching: each row is represented as a textual sequence (concatenated column name–cell value pairs), encoded into embeddings, and matched via 1-NN in the embedding space. The file comprises four columns: T1_index (task table row index), T2_index (matched auxiliary table row index), cosine_similarity, and cosine_distance. This mapping enables feature augmentation by concatenating matched auxiliary features to target records.

## References
