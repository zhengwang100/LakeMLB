## Overview

**London and Hong Kong Stocks (LHStocks)** is a tabular dataset designed for weakly related (Join-based) table scenarios in Data Lake(House) settings. The dataset focuses on publicly listed companies and comprises two weakly related tables: a task table (London and Hong Kong listed companies) and an auxiliary table (Wikipedia information). The task table (stocks.csv) contains information on 1,120 companies listed in London and Hong Kong, while the auxiliary table (wiki.csv) contains 937 records of company information available on Wikipedia. The objective of this dataset is to predict the industry sector of listed companies (a classification task).

The two tables exhibit a weak association (Join relationship), where information from the auxiliary table can be leveraged to enhance machine learning performance on the task table, thereby enabling more accurate relational analysis.

## Data Processing

The LHStocks dataset consists of two weakly related tables (a task table and an auxiliary table), derived from the official listings of London and Hong Kong exchanges and Wikipedia. The detailed data processing steps are described below:

**Task Table:** London and Hong Kong Listed Companies (stocks.csv)
- **Data Source:**
[London Stocks](https://docs.londonstockexchange.com/sites/default/files/reports/Issuer%20list_2.xlsx)
[Hong Kong Stocks](https://www3.hkexnews.hk/reports/dirsearch?sc_lang=en)
- **Data Cleaning:**
The two tables from London and Hong Kong were merged, with a Source column added to indicate the origin (London/Hong Kong) of each sample. Since the Industry classification system used in Hong Kong is highly similar to the ICB (Industry Classification Benchmark) system, the Hong Kong Industry categories were mapped to the ICB system to ensure label consistency after merging.
- **Data Filtering:**
Categories with more than 112 samples were retained based on the ICB Industry column (the Telecommunications category with 55 samples was excluded to ensure class balance and sufficiency). A total of 112 samples were randomly selected from each category (with priority given to samples that have corresponding Wikipedia information) to form the task table.

**Auxiliary Table:** Wikipedia Information (wiki.csv)
- **Data Cleaning:**
Infobox information was extracted from the Wikipedia homepage of corresponding companies, retaining only attributes with less than 95% missing values.
- **Data Filtering:**
Deduplication was performed when entity-matched Wikipedia pages were identical (e.g., parent companies and subsidiaries).

## Dataset Composition

- **stocks.csv**: This file contains information on 1,078 listed companies with 16 features, covering 10 industry sector categories.

- **wiki.csv**: This file contains 937 retrievable Wikipedia homepage infobox records, with 21 features after cleaning.

- **mapping.csv**: This file contains the mapping relationships obtained through 1NN matching based on entity names, which can be used for join operations to leverage feature augmentation from wiki.csv. It has a four-column structure: (T1_index, T2_index, cosine_similarity, cosine_distance).

<p>

- **mask.pt**: This file provides the train/validation/test split for the task table (stocks.csv). Since the dataset is class-balanced, samples within each category are randomly partitioned into train/validation/test sets at a ratio of 70%/10%/20%, respectively. The mask.pt file stores a dictionary containing three keys — train_mask, val_mask, and test_mask — each corresponding to a Boolean tensor of the same length as the total number of samples (1,078), indicating whether each sample belongs to the training, validation, or test set.

## References
