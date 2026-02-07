"""
TransTab contrastive learning script for (LHStocks) dataset.
Unsupervised pretraining.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', "lib"))

import transtab
import numpy as np
from sklearn.metrics import (roc_auc_score, classification_report, 
                            accuracy_score, precision_score, recall_score, f1_score)
from rllm.types import ColType


task_col_types = {
            "Source": ColType.CATEGORICAL,
            "Company Name": ColType.CATEGORICAL,
            "ICB Industry": ColType.CATEGORICAL,
            "Admission Date": ColType.CATEGORICAL,
            "Country of Incorporation": ColType.CATEGORICAL,
            "World Region": ColType.CATEGORICAL,
            "Market": ColType.CATEGORICAL,
            "International Issuer": ColType.CATEGORICAL,
            "Company Market Cap (￡m)": ColType.NUMERICAL,
            "Stock Code": ColType.CATEGORICAL,
            "Listing Status": ColType.CATEGORICAL,
            "Director's English Name": ColType.CATEGORICAL,
            "Capacity": ColType.CATEGORICAL,
            "Position": ColType.CATEGORICAL,
            "Appointment Date (yyyy-mm-dd)": ColType.CATEGORICAL,
            "Resignation Date (yyyy-mm-dd)": ColType.CATEGORICAL,
}

aux_col_types = {
            "wiki_title": ColType.CATEGORICAL,
            "wiki_url": ColType.CATEGORICAL,
            "area_served": ColType.CATEGORICAL,
            "company_type": ColType.CATEGORICAL,
            "founded": ColType.CATEGORICAL,
            "founders": ColType.CATEGORICAL,
            "headquarters": ColType.CATEGORICAL,
            "industry": ColType.CATEGORICAL,
            "key_people": ColType.CATEGORICAL,
            "net_income": ColType.CATEGORICAL,
            "num_employees": ColType.CATEGORICAL,
            "operating_income": ColType.CATEGORICAL,
            "owner": ColType.CATEGORICAL,
            "parent": ColType.CATEGORICAL,
            "products": ColType.CATEGORICAL,
            "revenue": ColType.CATEGORICAL,
            "subsidiaries": ColType.CATEGORICAL,
            "total_assets": ColType.CATEGORICAL,
            "total_equity": ColType.CATEGORICAL,
            "traded_as": ColType.CATEGORICAL,
            "website": ColType.CATEGORICAL,
}


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'table_lhstocks', 'raw'))
task_config = transtab.create_dataset_config(
    col_types_dict=task_col_types,
    target_col='ICB Industry', #target(task)
    mask_path=os.path.join(DATA_DIR, 'stocks_lh_mask.pt'),
)

# Auxiliary config for unsupervised contrastive learning
aux_config = transtab.create_dataset_config(
    col_types_dict=aux_col_types,
    target_col=None,  # No target(aux)
    mask_path=os.path.join(DATA_DIR, 'wiki_lh_mask.pt'),
)

print(f"Task config: {len(task_config['cat'])} cat, {len(task_config['num'])} num features")
print(f"Auxiliary config: {len(aux_config['cat'])} cat, {len(aux_config['num'])} num features")

# Stage 1: Contrastive Learning

allset1, trainset1, valset1, testset1, cat_cols1, num_cols1, bin_cols1 = transtab.load_data(
    [DATA_DIR], 
    dataset_config={DATA_DIR: aux_config}, 
    filename='wiki_lh.csv'
)

print(f"Train: {len(trainset1[0][0])}, Val: {len(valset1[0][0])}, Test: {len(testset1[0][0])}")

# Build contrastive learning model
model_pretrain, collate_fn = transtab.build_contrastive_learner(
    categorical_columns=cat_cols1,
    numerical_columns=num_cols1,
    binary_columns=bin_cols1,
    supervised=False,       # Unsupervised contrastive learning
    num_partition=4,        # Number of column partitions for pos/neg sampling
    overlap_ratio=0.5,      # Overlap ratio of column partitions during CL
)

# Train with contrastive learning
transtab.train(
    model_pretrain, 
    trainset1, 
    valset1, 
    collate_fn=collate_fn,
    num_epoch=100,
    lr=1e-4,
    eval_metric='val_loss',
    eval_less_is_better=True,
    output_dir='./ckpt_cl/pretrained'
)

print("Contrastive learning model saved to ./ckpt_cl/pretrained")

# Stage 2: Fine-tune

allset2, trainset2, valset2, testset2, cat_cols2, num_cols2, bin_cols2 = transtab.load_data(
    [DATA_DIR], 
    dataset_config={DATA_DIR: task_config}, 
    filename='stocks_lh.csv'
)

print(f"Train: {len(trainset2[0][0])}, Val: {len(valset2[0][0])}, Test: {len(testset2[0][0])}")

# Build classifier from pretrained checkpoint
model_downstream = transtab.build_classifier(
    categorical_columns=cat_cols2,
    numerical_columns=num_cols2,
    binary_columns=bin_cols2,
    num_class=10,
    checkpoint='./ckpt_cl/pretrained'  # Load pretrained weights
)

# Update model for new dataset
model_downstream.update({'cat': cat_cols2, 'num': num_cols2, 'bin': bin_cols2, 'num_class': 10})

# Fine-tune on Task table
transtab.train(
    model_downstream,
    trainset2,
    valset2,
    num_epoch=100,
    eval_metric='val_loss',
    eval_less_is_better=True,
    output_dir='./checkpoint'
)

x_test, y_test = testset2[0]
ypred_prob = transtab.predict(model_downstream, x_test, y_test)
auc_score = roc_auc_score(y_test, ypred_prob, multi_class='ovr')
accuracy = accuracy_score(y_test, np.argmax(ypred_prob, axis=1))
precision = precision_score(y_test, np.argmax(ypred_prob, axis=1), average='weighted')
recall = recall_score(y_test, np.argmax(ypred_prob, axis=1), average='weighted')
f1 = f1_score(y_test, np.argmax(ypred_prob, axis=1), average='weighted')

print(f'\nTest Performance:')
print(f'  AUC:       {auc_score:.4f}')
print(f'  Accuracy:  {accuracy:.4f}')
print(f'  Precision: {precision:.4f}')
print(f'  Recall:    {recall:.4f}')
print(f'  F1 Score:  {f1:.4f}')

print("\nClassification Report:")
print(classification_report(y_test, np.argmax(ypred_prob, axis=1), digits=4))

