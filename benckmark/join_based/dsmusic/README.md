## Overview

**Discogs and Spotify Music (DSMusic)** is a tabular dataset designed for weakly related (Join-based) table scenarios in Data Lake(House) settings. The dataset focuses on music tracks and comprises two weakly related tables: a task table (Discogs database) and an auxiliary table (Spotify tracks). The task table (Discogs_music.csv) contains 11,000 music track records from the publicly archived Discogs database, while the auxiliary table (Spotify_music.csv) contains 11,000 music track records from the Spotify music platform. The objective of this dataset is to predict music genres (a single-label multi-class classification task).

The two tables exhibit a weak association (Join relationship), where information from the auxiliary table can be leveraged to enhance machine learning performance on the task table, thereby enabling more accurate relational analysis.

## Data Processing

The DSMusic dataset consists of two weakly related tables (a task table and an auxiliary table), derived from the Discogs database and the Spotify online music platform. The detailed data processing steps are described below:

**Task Table:** Music from Discogs Database (Discogs_music.csv)
- **Data Source:** [Discogs_music](https://www.kaggle.com/datasets/fleshmetal/records-a-comprehensive-music-metadata-dataset)
- **Data Cleaning:**
The styles column was removed from the table (as it could potentially interfere with genre prediction).
- **Data Filtering:**
Single-label categories were extracted based on the genres column. A total of 1,000 samples were randomly drawn from each single-label category to form the task table, yielding 11 categories.

**Auxiliary Table:** Spotify Music Dataset (Spotify_music.csv)
- **Data Source:** [Spotify_music](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- **Data Filtering:**
1NN matching was performed based on the task table, retaining only samples that were matched to the task table.

## Dataset Composition

- **Discogs_music.csv**: This file contains 11,000 music track records from the Discogs database, with 5 features after cleaning, covering 11 music genre categories.

- **Spotify_music.csv**: This file contains 11,000 music track records with 20 features.

- **map.csv**: This file contains the mapping relationships obtained through 1NN matching based on track names, which can be used for join operations to leverage feature augmentation from Spotify_music.csv. It has a three-column structure: (T1_index, T2_index, cosine_similarity).

<p>

- **mask.pt**: This file provides the train/validation/test split for the task table (Discogs_music.csv). Since the dataset is class-balanced, samples within each category are randomly partitioned into train/validation/test sets at a ratio of 70%/10%/20%, respectively. The mask.pt file stores a dictionary containing three keys — train_mask, val_mask, and test_mask — each corresponding to a Boolean tensor of the same length as the total number of samples (11,000), indicating whether each sample belongs to the training, validation, or test set.

## References
