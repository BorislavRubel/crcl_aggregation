# ------------------------------------------------------------------------------ #
# -------------------------------- Imports ------------------------------------- #
# ------------------------------------------------------------------------------ #
import sys
import yaml
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path
from utilities import *

import warnings
warnings.filterwarnings("ignore")

print('Loading Aggregation Settings File...')
# ------------------------------------------------------------------------------ #
# ------------------- Load Aggregation Settings File --------------------------- #
# ------------------------------------------------------------------------------ #
p = argparse.ArgumentParser()
p.add_argument('--cfg', type=str, default=None, help='configuration yaml file to use')
args = vars(p.parse_args())
cfg_filepath = args['cfg']
cfg = load_yaml(cfg_filepath)

print('Seperating Variables into Aggregation Lists...')
# ----------------------------------------------------------------------------- #
# -------------------- Get Vitals/Meds/Labs for Aggregation ------------------- #
# ----------------------------------------------------------------------------- #
# Get lists of variables that will go into their own aggregation method
vtl_vars, med_vars, lab_vars = get_cols_to_aggregate(str(cfg['labeled_excel_file_dir']))

# Move and/or add variables to different lists as needed 
vtl_vars_augmented, med_vars_augmented, lab_vars_augmented = move_cols_to_new_agg_method(
                                        vtl_vars, med_vars, lab_vars,
                                        vitals_to_labs = cfg['vitals_to_labs'], vitals_to_meds = cfg['vitals_to_meds'], labs_to_vitals = cfg['labs_to_vitals'], 
                                        labs_to_meds = cfg['labs_to_meds'], meds_to_vitals = cfg['meds_to_vitals'], meds_to_labs = cfg['meds_to_labs'],
                                        add_vitals = cfg['add_vitals'], add_meds = cfg['add_meds'], add_labs = cfg['add_labs'])

# ----------------------------------------------------------------------------- #
# -------------- Set up Files and Directories for Aggregation ----------------- #
# ----------------------------------------------------------------------------- #

# set the path to the cohort filtered datasets
patient_cohort_data_dir = Path(str(cfg['primary_data_dir']))

# make sure the text set is not being read/looked at -----> Remove Eventually
hdf_train_val_files = [file for file in list(patient_cohort_data_dir.glob('*.hdf')) if ('test' not in str(file).lower())]

# create a new directory for saving aggregated results
aggregate_output_directory = Path(str(cfg['agg_output_dir']))
aggregate_output_directory.mkdir(parents = True, exist_ok = True)

print('Loading Training Sets...')
# ----------------------------------------------------------------------------- #
# ------------------ Load Training Sets for Mean Imputation ------------------- #
# ----------------------------------------------------------------------------- #
# calculate average summary statistics across all patients
train_picu_df_for_imputation = [f for f in hdf_train_val_files if ('train' in str(f).lower()) and ('picu' in str(f).lower())][0]
train_cticu_df_for_imputation = [f for f in hdf_train_val_files if ('train' in str(f).lower()) and ('cticu' in str(f).lower())][0]

# read in training sets from which to gather population means and other summary statistics
picu_train_df = pd.read_hdf(train_picu_df_for_imputation)
cticu_train_df = pd.read_hdf(train_cticu_df_for_imputation)

# # calculate value tables for filling empty values based on specified training set
# average_summary_df_picu = average_summary_matrix(picu_train_df)
# average_summary_df_cticu = average_summary_matrix(cticu_train_df)

# # initialize list of columns that will contain empty values based on lookback aggregation
# col_types_to_fill = ['Average', 'Median', 'Std', 'Max', 'Min', 'Count','_last_value' ]

print('Starting Aggregation...')
# ---------------------------------------------------------------------------- #
# ------------------ Run Aggregation and Format Files ------------------------ #
# ---------------------------------------------------------------------------- #

# set additional list of additional ORIGINAL columns to include in aggregated dataframes
include_cols_in_agg_df = cfg['original_cols_to_append_to_agg']

# loop over raw hdf files in cohort dataset
for file in hdf_train_val_files:

    # Create empty dataframe to append aggregate dataframes 
    all_patient_agg_df = pd.DataFrame()
    
    # load cohort data
    raw_df = pd.read_hdf(file)
    
    # forward fill and impute the cohort data with the means of the respective Icu
    if 'picu' in str(file).lower():
        imputed_df = impute_df(dataframe = raw_df, training_set = picu_train_df , cfg['impute_method'], cfg['arbitrary_impute_value'])
    elif 'cticu' in str(file).lower():
        imputed_df = impute_df(dataframe = raw_df, training_set = cticu_train_df , cfg['impute_method'], cfg['arbitrary_impute_value'])

    # create non-empty creatinine data and gather eids from NON IMPUTED DATAFRAME
    # since imputing will remove significance of taking non empty rows of creatinine
    non_empty_creatinine_df, cohort_eids = creat_eids_and_df(raw_df)

    # apply aggregate lookback function to each eid and append it to the main aggregation dataframe
    for crt_eid_i in tqdm(range(len(cohort_eids))):
        
        # apply the primary aggregation function
        agg_df = cr_lookback_aggregate_df(filtered_cohort_all_data = imputed_df, 
                                            creatinine_measurement_rows_df = non_empty_creatinine_df, 
                                            eid = cohort_eids[crt_eid_i],
                                            vital_vars_list = vtl_vars_augmented, 
                                            med_vars_list = med_vars_augmented,
                                            lab_vars_list = lab_vars_augmented,
                                            lookback_window = 24.0)

        # add the aggregated eid dataframe to the main data frame
        all_patient_agg_df = pd.concat([all_patient_agg_df, agg_df])
    
#     # Take the subset consisting of vital columns and lab columns since medication columns are binary and will not be empty
#     fill_na_cols = [c for c in all_patient_agg_df.columns if c.split('_')[-1] in col_types_to_fill or ('_last_value' in c)]

#     # Impute empty aggregate dataframe values with calculated training set means from respective icu training set 
#     if 'cticu' in str(file):
#         print(f'Filling {file} with Values from CTICU Training Set')
#         all_patient_agg_df[fill_na_cols] = all_patient_agg_df[fill_na_cols].apply(lambda x: fill_all_na(x, all_patient_agg_df, average_summary_df_picu), axis = 0)
#     else:  
#         print(f'Filling {file} with Values from PICU Training Set')
#         all_patient_agg_df[fill_na_cols] = all_patient_agg_df[fill_na_cols].apply(lambda x: fill_all_na(x, all_patient_agg_df, average_summary_df_cticu), axis = 0)
    
    # Add the creatinine truth column for model predictions, year/pid columns for kfold splitting
    agg_df_w_feats = add_feats_to_df(aggregated_df = all_patient_agg_df, 
                                     non_empty_filtered_df = non_empty_creatinine_df, 
                                     orig_feats_to_add_list = include_cols_in_agg_df)
    
    # save the results into a seperate directory
    output_file_path = aggregate_output_directory / (file.with_suffix('').name + '_agg.hdf')
    print('Exporting Aggregate DF to: ', output_file_path)
    agg_df_w_feats.to_hdf(output_file_path, key = 'agg_df')

print('* DONE *')



