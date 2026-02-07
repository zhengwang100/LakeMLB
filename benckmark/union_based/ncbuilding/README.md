## Overview

**New York and Chicago Building Violations (NCBuilding)** is a tabular dataset designed for weakly related (Union-based) table scenarios in Data Lake(House) settings. The dataset focuses on building violation complaint incidents and comprises two weakly related tables: a task table (New York complaint reports) and an auxiliary table (Chicago complaint reports). The task table (newyork.csv) contains 30,000 building violation complaint records from January 2023 to December 2024, while the auxiliary table (chicago.csv) contains 37,000 building violation complaint records from January 2023 to December 2024. The objective of this dataset is to predict the type of building violation being complained about (a classification task).

The two tables exhibit a weak association (Union relationship), where information from the auxiliary table can be leveraged to enhance machine learning performance on the task table, thereby enabling more accurate relational analysis.

## Data Processing

The NCBuilding dataset consists of two weakly related building violation complaint tables (a task table and an auxiliary table), derived from publicly available building violation complaint reports. The detailed data processing steps are described below:

**Task Table:** New York Complaint Reports (newyork.csv)
- **Data Source:** [Housing Maintenance Code Violations](https://catalog.data.gov/dataset/housing-maintenance-code-violations)
- **Data Cleaning and Filtering:**
Records from January 2023 to December 2024 were selected based on the InspectionDate column. Specific violation codes were extracted from the NOVDescription field for each record and stored in the StatuteCodes column; subsequently, multi-label samples were removed. To prevent label leakage, the OrderNumber and NOVDescription columns were removed, as these columns exhibited a nearly one-to-one correspondence with the label column. Categories with more than 1,000 samples in StatuteCodes were retained, and 1,000 samples were extracted from each category, yielding a total of 30,000 records for the task table.


**Auxiliary Table:** Chicago Complaint Reports (chicago.csv)
- **Data Source:** [Building Violations](https://catalog.data.gov/dataset/building-violations)
- **Data Cleaning and Filtering:**
To prevent label leakage, the VIOLATION CODE, DEPARTMENT BUREAU, and VIOLATION ORDINANCE columns were removed beforehand, as these columns exhibited a nearly one-to-one correspondence with the label column. Records from January 2023 to December 2024 were selected based on the VIOLATION LAST MODIFIED DATE column. Categories with more than 1,000 samples in the VIOLATION DESCRIPTION column were retained, and 1,000 samples were extracted from each category, yielding a total of 37,000 records for the auxiliary table.

## Dataset Composition
- **newyork.csv**: This file contains 30,000 building complaint records with 40 features, covering 30 building violation categories.

- **chicago.csv**: This file contains 37,000 building complaint records with 23 features, covering 37 building violation categories.

- **mask.pt**: This file provides the train/validation/test split for the task table (newyork.csv). Since the dataset is class-balanced, samples within each category are sorted chronologically (from oldest to newest) and partitioned into train/validation/test sets at a ratio of 70%/10%/20%, respectively. The mask.pt file stores a dictionary containing three keys — train_mask, val_mask, and test_mask — each corresponding to a Boolean tensor of the same length as the total number of samples (30,000), indicating whether each sample belongs to the training, validation, or test set.

## References
