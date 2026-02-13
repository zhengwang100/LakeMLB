"""
TransTab single-dataset training script for (MSTraffic) dataset.
Train and evaluate on Task dataset only (no transfer learning).
"""
import sys
import os
import argparse
# Add transtab parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', "lib"))

import transtab
import numpy as np
from sklearn.metrics import (roc_auc_score, classification_report, 
                            accuracy_score, precision_score, recall_score, f1_score)

# Import ColType for column type definitions (same format as rllm datasets)
from rllm.types import ColType

parser = argparse.ArgumentParser(description='TransTab Single Dataset Training')
parser.add_argument('--ckpt_dir', type=str, default='./checkpoint', 
                    help='Checkpoint directory for saving/loading models')
parser.add_argument('--num_epoch', type=int, default=100,
                    help='Number of training epochs')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='Device to use (cpu or cuda:0)')
args = parser.parse_args()

print(f"Arguments:")
print(f"  Checkpoint directory: {args.ckpt_dir}")
print(f"  Number of epochs: {args.num_epoch}")
print(f"  Device: {args.device}")

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
    # "Municipality": ColType.CATEGORICAL,  
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
    # combine cols(from aux table)
    "OBJECTID": ColType.NUMERICAL,
    "SE_ANNO_CAD_DATA": ColType.CATEGORICAL,
    "INCKEY": ColType.NUMERICAL,
    "COLDETKEY": ColType.NUMERICAL,
    "REPORTNO": ColType.CATEGORICAL,
    "STATUS": ColType.CATEGORICAL,
    "ADDRTYPE": ColType.CATEGORICAL,
    "INTKEY": ColType.NUMERICAL,
    "LOCATION": ColType.CATEGORICAL, #da
    "EXCEPTRSNCODE": ColType.NUMERICAL,
    "EXCEPTRSNDESC": ColType.NUMERICAL,
    "SEVERITYCODE": ColType.NUMERICAL,
    "SEVERITYDESC": ColType.CATEGORICAL,
    "COLLISIONTYPE": ColType.CATEGORICAL, #da
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
    "SDOT_COLCODE": ColType.NUMERICAL,
    "SDOT_COLDESC": ColType.CATEGORICAL,
    "INATTENTIONIND": ColType.CATEGORICAL,
    "UNDERINFL": ColType.CATEGORICAL,
    "WEATHER": ColType.CATEGORICAL, #da
    "ROADCOND": ColType.CATEGORICAL,
    "LIGHTCOND": ColType.CATEGORICAL,
    "DIAGRAMLINK": ColType.CATEGORICAL,
    "REPORTLINK": ColType.CATEGORICAL,
    "PEDROWNOTGRNT": ColType.NUMERICAL,
    "SDOTCOLNUM": ColType.NUMERICAL,
    "SPEEDING": ColType.NUMERICAL,
    "STCOLCODE": ColType.NUMERICAL,
    "ST_COLDESC": ColType.CATEGORICAL,
    "SEGLANEKEY": ColType.NUMERICAL,
    "CROSSWALKKEY": ColType.NUMERICAL,
    "HITPARKEDCAR": ColType.CATEGORICAL,
    "SPDCASENO": ColType.CATEGORICAL,
    "Source of the collision report": ColType.CATEGORICAL,
    "Source description": ColType.CATEGORICAL,
    "Added date": ColType.CATEGORICAL,
    "Modified date": ColType.CATEGORICAL,
    "SHAREDMICROMOBILITYCD": ColType.NUMERICAL,
    "SHAREDMICROMOBILITYDESC": ColType.NUMERICAL,
    "x": ColType.NUMERICAL,
    "y": ColType.NUMERICAL,
}

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'table_mstraffic', 'raw'))
task_config = transtab.create_dataset_config(
    col_types_dict=task_col_types,
    target_col='Collision Type',
    mask_path=os.path.join(DATA_DIR, 'maryland_mask.pt'),
)

print(f"Task config: {len(task_config['cat'])} cat, {len(task_config['num'])} num features")

# Load Data
allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = transtab.load_data(
    [DATA_DIR], 
    dataset_config={DATA_DIR: task_config}, 
    filename='maryland.csv'
)

print(f"Train: {len(trainset[0][0])}, Val: {len(valset[0][0])}, Test: {len(testset[0][0])}")


# Build classifier
model = transtab.build_classifier(
    categorical_columns=cat_cols,
    numerical_columns=num_cols,
    binary_columns=bin_cols,
    num_class=9,
    num_layer=4,
    device=args.device
)

# Train
transtab.train(model, trainset, valset, num_epoch=args.num_epoch, eval_metric='val_loss',
               eval_less_is_better=True, output_dir=args.ckpt_dir)

# Load best model
model.load(args.ckpt_dir)
print(f"Loaded best model from {args.ckpt_dir}")

x_test, y_test = testset[0]
ypred_prob = transtab.predict(model, x_test, y_test)
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
