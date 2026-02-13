"""
TransTab training script for (MSTraffic) dataset.
Transfer learning: pretrain on Auxiliary table (Seattle), fine-tune on Task table (Maryland).
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', "lib"))

import transtab
import numpy as np
from sklearn.metrics import (roc_auc_score, classification_report, 
                            accuracy_score, precision_score, recall_score, f1_score)
from rllm.types import ColType

# ==================== Parse Arguments ====================

parser = argparse.ArgumentParser(description='TransTab Transfer Learning')
parser.add_argument('--ckpt_dir', type=str, default='./checkpoint',
                    help='Temporary checkpoint directory during training')
parser.add_argument('--pretrain_dir', type=str, default='./ckpt_transfer/pretrained',
                    help='Directory for saving/loading pretrained model')
parser.add_argument('--num_epoch_pretrain', type=int, default=1,
                    help='Number of epochs for pretraining')
parser.add_argument('--num_epoch_finetune', type=int, default=1,
                    help='Number of epochs for fine-tuning')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='Device to use (cpu or cuda:0)')
args = parser.parse_args()

print(f"Arguments:")
print(f"  Checkpoint directory: {args.ckpt_dir}")
print(f"  Pretrain directory: {args.pretrain_dir}")
print(f"  Pretrain epochs: {args.num_epoch_pretrain}")
print(f"  Finetune epochs: {args.num_epoch_finetune}")
print(f"  Device: {args.device}")

# Task table column types
task_col_types = {
    "Report Number": ColType.CATEGORICAL,
    "Local Case Number": ColType.CATEGORICAL,
    "Agency Name": ColType.CATEGORICAL,
    "ACRS Report Type": ColType.CATEGORICAL,
    "Crash Date/Time": ColType.CATEGORICAL,
    "Hit/Run": ColType.CATEGORICAL,
    "Route Type": ColType.CATEGORICAL,
    "Lane Direction": ColType.CATEGORICAL,
    "Lane Type": ColType.CATEGORICAL,
    "Number of Lanes": ColType.CATEGORICAL,
    "Direction": ColType.CATEGORICAL,
    "Distance": ColType.NUMERICAL,
    "Distance Unit": ColType.CATEGORICAL,
    "Road Grade": ColType.CATEGORICAL,
    "Road Name": ColType.CATEGORICAL,
    "Cross-Street Name": ColType.CATEGORICAL,
    "Off-Road Description": ColType.CATEGORICAL,
    "Related Non-Motorist": ColType.CATEGORICAL,
    "At Fault": ColType.CATEGORICAL,
    "Collision Type": ColType.CATEGORICAL,
    "Weather": ColType.CATEGORICAL,
    "Surface Condition": ColType.CATEGORICAL,
    "Light": ColType.CATEGORICAL,
    "Traffic Control": ColType.CATEGORICAL,
    "Driver Substance Abuse": ColType.CATEGORICAL,
    "Non-Motorist Substance Abuse": ColType.CATEGORICAL,
    "First Harmful Event": ColType.CATEGORICAL,
    "Second Harmful Event": ColType.CATEGORICAL,
    "Junction": ColType.CATEGORICAL,
    "Intersection Type": ColType.CATEGORICAL,
    "Road Alignment": ColType.CATEGORICAL,
    "Road Condition": ColType.CATEGORICAL,
    "Road Division": ColType.CATEGORICAL,
    "Latitude": ColType.NUMERICAL,
    "Longitude": ColType.NUMERICAL,
    "Location": ColType.CATEGORICAL,
}

# Auxiliary table column types
aux_col_types = {
    "OBJECTID": ColType.NUMERICAL,
    "SE_ANNO_CAD_DATA": ColType.CATEGORICAL,
    "INCKEY": ColType.NUMERICAL,
    "COLDETKEY": ColType.NUMERICAL,
    "REPORTNO": ColType.CATEGORICAL,
    "STATUS": ColType.CATEGORICAL,
    "ADDRTYPE": ColType.CATEGORICAL,
    "INTKEY": ColType.NUMERICAL,
    "LOCATION": ColType.CATEGORICAL,
    "SEVERITYCODE": ColType.CATEGORICAL,
    "SEVERITYDESC": ColType.CATEGORICAL,
    "COLLISIONTYPE": ColType.CATEGORICAL,
    "PERSONCOUNT": ColType.NUMERICAL,
    "PEDCOUNT": ColType.NUMERICAL,
    "PEDCYLCOUNT": ColType.NUMERICAL,
    "VEHCOUNT": ColType.NUMERICAL,
    "INJURIES": ColType.NUMERICAL,
    "SERIOUSINJURIES": ColType.NUMERICAL,
    "FATALITIES": ColType.NUMERICAL,
    "INCDATE": ColType.CATEGORICAL,
    "INCDTTM": ColType.CATEGORICAL,
    "JUNCTIONTYPE": ColType.CATEGORICAL,
    "UNDERINFL": ColType.CATEGORICAL,
    "WEATHER": ColType.CATEGORICAL,
    "ROADCOND": ColType.CATEGORICAL,
    "LIGHTCOND": ColType.CATEGORICAL,
    "DIAGRAMLINK": ColType.CATEGORICAL,
    "REPORTLINK": ColType.CATEGORICAL,
    "PEDROWNOTGRNT": ColType.CATEGORICAL,
    "SPEEDING": ColType.CATEGORICAL,
    "CROSSWALKKEY": ColType.NUMERICAL,
    "HITPARKEDCAR": ColType.CATEGORICAL,
    "SPDCASENO": ColType.CATEGORICAL,
    "Source of the collision report": ColType.CATEGORICAL,
    "Source description": ColType.CATEGORICAL,
    "Added date": ColType.CATEGORICAL,
    "Modified date": ColType.CATEGORICAL,
    "SHAREDMICROMOBILITYCD": ColType.CATEGORICAL,
    "SHAREDMICROMOBILITYDESC": ColType.CATEGORICAL,
    "x": ColType.NUMERICAL,
    "y": ColType.NUMERICAL,
}

# ==================== Configuration Generation ====================

# Data directory (absolute path)
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'table_mstraffic', 'raw'))

# Create dataset configs
task_config = transtab.create_dataset_config(
    col_types_dict=task_col_types,
    target_col='Collision Type',
    mask_path=os.path.join(DATA_DIR, 'maryland_mask.pt'),
)

aux_config = transtab.create_dataset_config(
    col_types_dict=aux_col_types,
    target_col='COLLISIONTYPE',
    mask_path=os.path.join(DATA_DIR, 'mask_seattle.pt'),
)

print(f"Task config: {len(task_config['cat'])} cat, {len(task_config['num'])} num features")
print(f"Auxiliary config: {len(aux_config['cat'])} cat, {len(aux_config['num'])} num features")

# Pretrain on Auxiliary table
print("Stage 1: Pretraining")

# Load Seattle data
allset1, trainset1, valset1, testset1, cat_cols1, num_cols1, bin_cols1 = transtab.load_data(
    [DATA_DIR], 
    dataset_config={DATA_DIR: aux_config}, 
    filename='seattle.csv'
)

print(f"Train: {len(trainset1[0][0])}, Val: {len(valset1[0][0])}, Test: {len(testset1[0][0])}")

# Build classifier
model = transtab.build_classifier(
    categorical_columns=cat_cols1,
    numerical_columns=num_cols1,
    binary_columns=bin_cols1,
    num_class=9,
    num_layer=4,
    device=args.device
)

# Train
transtab.train(model, trainset1, valset1, num_epoch=args.num_epoch_pretrain, eval_metric='val_loss',
               eval_less_is_better=True, output_dir=args.ckpt_dir)

# Save model
model.save(args.pretrain_dir)
print(f"Model saved to {args.pretrain_dir}")

# Fine-tune on Task table

# Load pretrained model
model.load(args.pretrain_dir)

# Load Maryland data
allset2, trainset2, valset2, testset2, cat_cols2, num_cols2, bin_cols2 = transtab.load_data(
    [DATA_DIR], 
    dataset_config={DATA_DIR: task_config}, 
    filename='maryland.csv'
)

print(f"Train: {len(trainset2[0][0])}, Val: {len(valset2[0][0])}, Test: {len(testset2[0][0])}")

# Update model for new dataset
model.update({'cat': cat_cols2, 'num': num_cols2, 'bin': bin_cols2, 'num_class': 9})

# Fine-tune
transtab.train(model, trainset2, valset2, num_epoch=args.num_epoch_finetune, eval_metric='val_loss',
               eval_less_is_better=True, output_dir=args.ckpt_dir)

# Predict
x_test, y_test = testset2[0]
ypred_prob = transtab.predict(model, x_test, y_test)

# Calculate metrics
auc_score = roc_auc_score(y_test, ypred_prob, multi_class='ovr')
accuracy = accuracy_score(y_test, np.argmax(ypred_prob, axis=1))
precision = precision_score(y_test, np.argmax(ypred_prob, axis=1), average='weighted')
recall = recall_score(y_test, np.argmax(ypred_prob, axis=1), average='weighted')
f1 = f1_score(y_test, np.argmax(ypred_prob, axis=1), average='weighted')

# Print results
print(f'\nTest Performance:')
print(f'  AUC:       {auc_score:.4f}')
print(f'  Accuracy:  {accuracy:.4f}')
print(f'  Precision: {precision:.4f}')
print(f'  Recall:    {recall:.4f}')
print(f'  F1 Score:  {f1:.4f}')

print("\nClassification Report:")
print(classification_report(y_test, np.argmax(ypred_prob, axis=1), digits=4))

print("="*70)
print("Training and evaluation completed!")
print("="*70)
