import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import yaml

# ------------------------------------------------------------------------------------------------------------------------------------------# 
# ------------------------------------------------------------------------------------------------------------------------------------------#
def load_yaml(filepath):
    '''Loads a yaml from a filepath
    '''
    
    with open(filepath, 'r') as fin:
        ret = yaml.load(fin, Loader=yaml.FullLoader)
    return ret

# ------------------------------------------------------------------------------------------------------------------------------------------# 
# ------------------------------------------------------------------------------------------------------------------------------------------#
def get_cols_to_aggregate(discard_labeled_file_path, category_split_col_name = 'categ', discard_col_name = 'Discard'):
    '''
    Description:
        Reads in an excel file containing column information for what variables will be aggregated in 24 hour windows leading up to creatinine recordings
    Input:
        discard_labeled_file_path [Path]: 
        category_split_col_name [string]: 
        discard_col_name [string]:
    Output:
    '''
    
    labeled_variables_df = pd.read_excel(discard_labeled_file_path, engine='openpyxl')

    # Get medication and vital columns that neely marked as ** DO NOT ** discard
    vtl_variables_neely = list(labeled_variables_df[(labeled_variables_df[category_split_col_name] == 'Vitals') & (labeled_variables_df[discard_col_name] == 0.0)]['Unnamed: 0'])
    med_variables_neely = list(labeled_variables_df[(labeled_variables_df[category_split_col_name] == 'Drugs') & (labeled_variables_df[discard_col_name] == 0.0)]['Unnamed: 0'])
    lab_variables_neely = list(labeled_variables_df[(labeled_variables_df[category_split_col_name] == 'Labs') & (labeled_variables_df[discard_col_name] == 0.0)]['Unnamed: 0'])
    
    # prompt the user
    print('-------------')
    print(f'{len(vtl_variables_neely)} Vitals, {len(med_variables_neely)} Meds, and {len(lab_variables_neely)} Labs as Do Not Discard.')

    # display information about medication protocols among medication variables
    inter_meds = len([ i for i in med_variables_neely if '_inter' in i.lower()])
    cont_meds = len([ i for i in med_variables_neely if '_cont' in i.lower()])
    
    # prompt the user
    print(f'Neely Labeled Inter {inter_meds} Meds and {cont_meds} Cont Meds for potential use.')
    print('-------------')

    # return the list of variables that will be iterated over
    return vtl_variables_neely, med_variables_neely, lab_variables_neely

# ------------------------------------------------------------------------------------------------------------------------------------------# 
# ------------------------------------------------------------------------------------------------------------------------------------------#
def move_cols_to_new_agg_method(vtls_col_list, meds_col_list, labs_col_list, 
                                vitals_to_labs = [], vitals_to_meds = [], labs_to_vitals = [], 
                                labs_to_meds = [], meds_to_vitals = [], meds_to_labs = [],
                                add_vitals = [], add_meds = [], add_labs = []):
    '''
    Description:
        Allows the user to move variables to different aggregation methods when constructing primary aggregation dataframes.
        Can add columns that should be included in aggregation or reorganize existing aggregation columns into another procedure. 
        For example, moving X from vitals will remove it from vitals list and add it to either labs or meds as specified by user
    Input:
        vtls_col_list [list]: the list of vitals labeled as do not discard from the Neely excel file
        meds_col_list [list]: the list of medications labeled as do not discard from the Neely excel file
        labs_col_list [list]: the list of labs labeled as do not discard from the Neely excel file
        vitals_to_labs [list]: the list of vitals to be removed from vitals and added to labs
        vitals_to_meds [list]: the list of vitals to be removed from vitals and added to medications
        labs_to_vitals [list]: the list of labs to be removed from labs and added to vitals
        labs_to_meds [list]: the list of labs to be removed from labs and added to meds
        meds_to_vitals [list]: the list of medications to be removed from medications and added to vitals
        meds_to_labs [list]: the list of medications to be removed from medications and added to labs
        add_vitals [list]: the list of new vitals to be added to vitals
        add_meds [list]: the list of new medications to be added to medications
        add_labs [list]: the list of new labs to be added to labs
    Output:
        vtl_vars_augmented [list]: the reorganized vitals list that will be used for aggregation
        med_vars_augmented [list]: the reorganized medications list that will be used for aggregation
        lab_vars_augmented [list]: the reorganized labs list that will be used for aggregation
    '''
    
    print('Original List Lengths:')
    print(f'Vitals Variables: {len(vtls_col_list)}')
    print(f'Meds Variables: {len(meds_col_list)}')
    print(f'Labs Variables: {len(labs_col_list)}')
    print('-----------------')
    
    # remove variables from respective column types and make sure creatinine is not being aggregated
    vtl_vars = [vit for vit in vtls_col_list if vit not in (vitals_to_labs + vitals_to_meds + ['Creatinine'])]
    med_vars = [med for med in meds_col_list if med not in (meds_to_vitals + meds_to_labs + ['Creatinine'])]
    lab_vars = [lab for lab in labs_col_list if lab not in (labs_to_vitals + labs_to_meds + ['Creatinine'])]
    
    # add removed variables to different aggregation lists
    vtl_vars_augmented = list(set(vtl_vars + (meds_to_vitals + labs_to_vitals + add_vitals)))
    med_vars_augmented = list(set(med_vars + (vitals_to_meds + labs_to_meds + add_meds)))
    lab_vars_augmented = list(set(lab_vars + (vitals_to_labs + meds_to_labs + add_labs)))
    print('Lists With New Variables Added Lengths:')
    print(f'Vitals Variables: {len(vtl_vars_augmented)}')
    print(f'Meds Variables: {len(med_vars_augmented)}')
    print(f'Labs Variables: {len(lab_vars_augmented)}')
    print('-----------------')

    # return the new columns for aggregation 
    return vtl_vars_augmented, med_vars_augmented, lab_vars_augmented

# ------------------------------------------------------------------------------------------------------------------------------------------# 
# ------------------------------------------------------------------------------------------------------------------------------------------# 
def forward_fill (col): 
    '''
    Description:
        Given a column, forward fill the empty values. This function is for applying ffill on a per patient basis w pd.apply()
    Input:
        col [pd.Series]: the column that will be forward filled
    Output:
        col.ffill() [pd.Series] : the given column now forward fill
    '''
    return col.ffill()

# ------------------------------------------------------------------------------------------------------------------------------------------# 
# ------------------------------------------------------------------------------------------------------------------------------------------# 
def impute_df(dataframe, training_set, imputation_method, impute_value):
    '''
    '''
    
    # Initially forward fill patients on a per patient, basis using the forward fill function
    dataframe_ffill =  dataframe.groupby(level = 0).apply(lambda x: forward_fill(x))
    
    # empty values will remain from forward fill, so need to fill the rest with a second method
    if imputation_method == 'arbitrary value':
        
        # fill all empty values with an arbitrary/impossible value
        dataframe_filled = dataframe_ffill.fillna(impute_value)
    
    if imputation_method == 'training set mean':
        
        # fill every column by mapping corresponding training set means to the empty values
        for col in dataframe.columns:
            dataframe_ffill[col] = dataframe_ffill[col].fillna(training_set.mean()[col])
        
        # set the new dataframe to a variable to maintain return value naming convention
        dataframe_filled = dataframe_ffill
        
    # return the imputed dataframe
    return dataframe_filled
    
# ------------------------------------------------------------------------------------------------------------------------------------------# 
# ------------------------------------------------------------------------------------------------------------------------------------------# 
def creat_eids_and_df(raw_time_series_df, look_ahead = 48.0):
    '''
    Description:
        Takes only the rows with non-empty creatinine readings from a raw input dataset and calculates the creatinine 48-hours-later
        truth column for those selected rows. Returns a list of eids for iterating through the primary dataset in later preprocessing steps
    Input:
        primary_ds22_data_set [pd.DataFrame]: the original train/validation set containing all original information from our database
    Output:
        cohort_df [pd.DataFrame]: the original dataframe filtered to contain only eids longer than 48 hours
        df_nonempty_creatinine [pd.DataFrame]: the original dataframe filtered to contain only rows with non empty creatinine recordings 
    '''
    
    # create a future time column in the raw dataset
    time_col_name = 'absoluteTime'
    future_time_col_name = 'lookaheadTime'
    raw_time_series_df[future_time_col_name] = raw_time_series_df[time_col_name] + look_ahead
    
    # initialize creatinine column name for tracking non-empy column of interest
    non_empty_var_name = 'Creatinine'
    
    # create an empty truth column in the cohort-filtered raw dataset
    truth_col = non_empty_var_name + '_last_48_hours'
    raw_time_series_df[truth_col] = np.nan
    
    # Loop through each episode in dataset
    for eid, eid_data in raw_time_series_df[[time_col_name, non_empty_var_name]].groupby(level=0):

        # Grab the rows containing actual measurements of creatinine
        notNulls = eid_data[~eid_data[non_empty_var_name].isnull()]

        # construct look-ahead times of the temporal grid points of this eid
        future_t_points = raw_time_series_df[future_time_col_name].loc[eid]

        # Use linear interpolation to get corresponding values at those future times
        raw_time_series_df[truth_col].loc[eid] = np.interp(future_t_points, notNulls[time_col_name], notNulls[non_empty_var_name])
    
    # take all rows where creatinine is nonempty
    df_nonempty_creatinine = raw_time_series_df[~raw_time_series_df[non_empty_var_name].isnull()]
    non_empty_creatinine_eids = df_nonempty_creatinine.index.get_level_values(0).unique()
    
    # return the dataframe containing 48hr+ hour stays and the rows containing non-empty creatinine rows
    return df_nonempty_creatinine, non_empty_creatinine_eids

# ------------------------------------------------------------------------------------------------------------------------------------------# 
# ------------------------------------------------------------------------------------------------------------------------------------------# 
def cr_lookback_aggregate_df(filtered_cohort_all_data, creatinine_measurement_rows_df, eid, vital_vars_list, med_vars_list, lab_vars_list, lookback_window = 24.0):
    '''
    Description:
        Applies an aggregate method, containing mean, median, and standard deviation, on each variable of interest, 
        relative to the 24 hour lookback window given a creatinine recording. Then applies a binarizing condition
        on all given medication columns, aoolying a 1 if any version of the medicaiton was administered in the 24 hour
        lookback period. Every variable will have 3 additional aggregate columns and a binarized medication column. 
        This is applied to one eid at a time and can be looped over the primary cohort data frame.
    
    Input:
        filtered_cohort_all_data [pd.DataFrame]: The primary DS22 dataset containing only eids longer than 48 hours
        creatinine_measurement_rows_df [pd.DataFrame]: dataframe containing rows with recorded creatinine measurements for every eid in cohort
        eid [int]: the encounter id number which must be a patient 
        vital_vars_list [list]: list of variables contained in ds22, as stated for use by Neely Group 
        med_vars_list [list]: list of medication column names contained in ds22, as stated for use by Neely Group
        lab_vars_list [list]: list of lab column names contained in ds22, as stated for use by Neely Group
        
    Output:
        non_empty_creatinine_data [pd.DataFrme]:DataFrame containing aggregated and binarized med info for every non empty
                                                recorded creatinine row, as given by looking back 24hrs at each row
    '''

    # -------------------------------------------- Set Up Variables for Recording Information --------------------------------------- # 
    # create a comprehensive list containing all the columns of interest that will be aggregated
    keep_original_cols =  ['absoluteTime'] + ['Creatinine'] + ['Creatinine_Truth']
    cols_to_aggregate = vital_vars_list + med_vars_list + lab_vars_list
    primary_cols_list = keep_original_cols + cols_to_aggregate
        
    # filter the data sets on the chosen eid value and take all needed columns relative
    all_data_dataframe = filtered_cohort_all_data.loc[eid, primary_cols_list]
    non_empty_creatinine_data = creatinine_measurement_rows_df.loc[eid, primary_cols_list]
        
    # calculate the number of new rows that will be created in the dataframe generation
    creat_n_rows = len(non_empty_creatinine_data)

    # ----------------------------- get list of unique medications  --------------------------- #
    inter_meds_unique = [md.split('_inter')[0] for md in med_vars_list if ('_int' in md)]
    cont_meds_unique = [md.split('_cont')[0] for md in med_vars_list if ('_cont' in md)]
    unique_inter_cont_meds = list(pd.Series(inter_meds_unique + cont_meds_unique).unique())
    # get list of other variables that will follow medication aggregation protocol
    other_meds = [md.split(' ')[0] for md in med_vars_list if ('_inter' not in md) and ('_cont' not in md)]
    other_meds_unique = list(pd.Series(other_meds).unique())
    # combine the two lists of medication subtypes
    all_unique_meds = unique_inter_cont_meds + other_meds_unique
    
    # create empty dictionary for recording aggregations for unique medications 
    binarized_meds_dict = {}
    for med_name in all_unique_meds:
        binarized_meds_dict[med_name] = [0] * creat_n_rows

    # construct empty dicationary for calculating summary statistics on every column in the data frame
    summary_stats_keys_vitals = ['Average', 'Median', 'Std', 'Max', 'Min', 'Count']
    summary_statistics_dict = {}
    for column in vital_vars_list:
        summary_statistics_dict[column] = {sum_stat : [0] * creat_n_rows for sum_stat in summary_stats_keys_vitals} 
            
    # construct dictionary for recording last lab results
    last_lab_vals_dict = {}
    for lab in lab_vars_list:
        last_lab_vals_dict[lab] = [0] * creat_n_rows
            
    # intiailize empty arrays for recording interval information
    prev_24_start = []
    prev_24_end = []
        
    # Set the dictionary index that will be incremented in order to update location in each empty summary dicationary
    dictionary_index = 0
        
    # -------------------------------------------- Iterate and Collect Column Information --------------------------------------- # 
    # iterate through all non empty creatinine rows and create a summary row for each 
    # for dictionary_index, (index, data) in enumerate(non_empty_creatinine_data.iterrows()): EL 
    for index, data in non_empty_creatinine_data.iterrows():

        # construct lookback window bounds and set end bound to 0 if it is negative (out of bounds)
        start = data['absoluteTime']
        end = max([0, data['absoluteTime'] - lookback_window])
                            
        # add lookback interval information to respective columns
        prev_24_start.append(start)
        prev_24_end.append(end)

        # filter the data frame from the main data set contained within the lookback window
        lookback_24_df = all_data_dataframe[(all_data_dataframe['absoluteTime'] <= start) & (all_data_dataframe['absoluteTime'] >= end)]

        # -------------------------------------------- Binarize Medication Protocols --------------------------------------- # 
        #binarize every medications column in the medication columns list
        for med in all_unique_meds:

            # determine if any variation of the medication type was admininstered during the lookback window
            # 1) take all columns containing the medication string
            cols_w_med = [col for col in med_vars_list if (med in col.split('_inter')) or (med in col.split('_cont')) or (med in col.split(' '))]
            current_med_df = lookback_24_df[cols_w_med]
                
            # 2) take the sum of all columns containing the medication
            meds_administered = current_med_df.sum()
                
            # 3) take the cumilative sum of the sum to check if it is nonzero                
            binarized_meds_dict[med][dictionary_index] = int(sum(meds_administered) > 0)
            
        # --------------------------------------------- Summarize Vital Variables ------------------------------------------ # 
        # calculate summary statistics for each ***NON MED/DRUG/LABS*** column and place them into the summary dictonary
        column_statistics = lookback_24_df.describe()
        for col_key in list(summary_statistics_dict.keys()):
            summary_statistics_dict[col_key]['Average'][dictionary_index] = column_statistics[col_key].loc['mean']
            summary_statistics_dict[col_key]['Median'][dictionary_index] = column_statistics[col_key].loc['50%']
            summary_statistics_dict[col_key]['Std'][dictionary_index] = column_statistics[col_key].loc['std']
            summary_statistics_dict[col_key]['Max'][dictionary_index] = column_statistics[col_key].loc['max']
            summary_statistics_dict[col_key]['Min'][dictionary_index] = column_statistics[col_key].loc['min']
            summary_statistics_dict[col_key]['Count'][dictionary_index] = column_statistics[col_key].loc['count']
            
        # --------------------------------------------- Get Last Valid Lab Values ------------------------------------------ #
        # find the index of the last non-nan lab value for each lab
        last_valid_result_per_lab = lookback_24_df.apply(pd.Series.last_valid_index)
         
        # iterate through every lab in the lab variables list
        for lab in lab_vars_list:
                
            # get the index of the last lab result for the currently iterated lab variable
            last_lab_index = last_valid_result_per_lab[lab]
                
            # check if a non-empty/valid value exists in the lookback window
            if math.isnan(last_lab_index) == False:
                    
                # convert the index to an integer in order to index into the lookback window 
                last_lab_index = int(last_lab_index) 
                    
                # take the value corresponding to the last recorded lab result
                last_lab_value_in_lookback = lookback_24_df.loc[last_lab_index][lab]
                    
                # record the latest lab value in the lookback window into the labs dictionary 
                last_lab_vals_dict[lab][dictionary_index] = last_lab_value_in_lookback
                
            # if a non-empty lab result does not exist, set the last lab result as nan
            else:
                last_lab_vals_dict[lab][dictionary_index] = np.nan
            
        # update the dictionary index in order to update next empty value relative to the next non-empty creatinine row
        dictionary_index = dictionary_index + 1 

    # --------------------------------------------- Construct Columns from Preproccessing Functions ------------------------------------------ # 
    # create interval start column
    non_empty_creatinine_data['Prev 24hrs Lookback Start'] = prev_24_start

    # create interval end column and adjust to 0 if it is negative 
    non_empty_creatinine_data['Prev 24hrs Lookback End'] = prev_24_end
        
    # add eid as column for later indexing
    non_empty_creatinine_data['EID'] = eid 
        
    # create individual column for every summary statistic in the summary dictionary 
    for summarized_column in summary_statistics_dict:
        for statistic in summary_stats_keys_vitals:
            non_empty_creatinine_data[summarized_column + '_' + statistic] = summary_statistics_dict[summarized_column][statistic]

    # create individual column for every binarized medication in the medications dictionary
    for binary_med in binarized_meds_dict:
        non_empty_creatinine_data[binary_med + '_present_binary'] = binarized_meds_dict[binary_med]
            
     # create individual column for every binarized medication in the medications dictionary
    for lab_name in last_lab_vals_dict:
         non_empty_creatinine_data[lab_name + '_last_value'] = last_lab_vals_dict[lab_name]

    # --------------------------------------------- Edit Results and Return Proper Format ------------------------------------------ # 
    # return only the newly computed columns, including the creatinine measurement and timesteps
    num_cols_to_remove = len(primary_cols_list)
    only_new_cols_df = non_empty_creatinine_data.iloc[:, num_cols_to_remove:]
    only_new_cols_df['Creatinine'] = non_empty_creatinine_data['Creatinine']
    only_new_cols_df['Timestep'] = non_empty_creatinine_data.index.get_level_values(0).unique()
        
    # set the eid column and timestamp column as values for multi-indexing
    only_new_cols_df = only_new_cols_df.set_index(['EID','Timestep'])
        
    # return final result 
    return only_new_cols_df

# ------------------------------------------------------------------------------------------------------------------------------------------# 
# ------------------------------------------------------------------------------------------------------------------------------------------# 
def add_feats_to_df(aggregated_df, non_empty_filtered_df, orig_feats_to_add_list):
    '''
    Description:
        Adds additional features to the pre-generated aggregated datasets for utility/training purposes
    Input:
        aggregated_df [pd.DataFrame]: the aggregated dataframe containing only rows of non-empty creatinine data, relative to lookback windows
        non_empty_filtered_df [pd.DataFrame]: a dataframe containing only the non-empty creatinine rows relative to the raw, cohort filtered datasets
        orig_feats_to_add_list [list]: a list of features (column names) from the original dataset that could be added to the aggregated dataframe
    
    Output:
        added_feats_df [pd.DataFrame]: the creatinine-lookback, aggregated dataframe with additional features added as individual columns
    '''
    
    # check if any of the features to be added are already in the dataframes columns
#     pd.Series(['A','B']).isin(raw_df.columns).all()
    
    # if the length of the column being added matches the length of the aggregated dataframe, append the desired features
    if len(non_empty_filtered_df) == len(aggregated_df):
        
        # construct a dataframe of ORIGINAL features that will be added to the aggregated dataset as columns
        df_to_merge = non_empty_filtered_df[orig_feats_to_add_list]

        # combine the aggregated data frame with the dataframe containing additional features of interest
        added_feats_df = pd.concat([aggregated_df, df_to_merge], axis = 1)

        # return the data aggregated dataframe with additional columns containing original data
        return added_feats_df
    
    # if incompatible dataframes are given, prompt the user with an error
    else:
        print('Incorrect Dimensions for Adding Additional Features!')
        return 0
# ------------------------------------------------------------------------------------------------------------------------------------------# 
# ------------------------------------------------------------------------------------------------------------------------------------------#       
def average_count(col, time_period = 24.0):
    '''
    Description:
        Calculates the average number of recordings in a 24 hour period for a given column
    Input:
    Output:
    '''
    
    # Find the total number of time points in the column
    total_time_points = len(col)
    
    # find the number of non-empty recorded values in the column
    number_of_valid_recordings = col.count()

    # calculate the average number of recordings in a 24 hour period 
    average_recording_count = (number_of_valid_recordings / total_time_points) * time_period
    
    # return the average 
    return average_recording_count
    
# ------------------------------------------------------------------------------------------------------------------------------------------# 
# ------------------------------------------------------------------------------------------------------------------------------------------#   
def average_summary_matrix(icu_type_training_data): 
    '''
    Deccription:
        Calculates the AVERAGE mean, median, standard deviation, max, min, and count across all patients in a given training set.
        Stores the information in a dataframe (matrix) format where the columns are summary statistics and the rows are the 
        column names on which the averaging was performed. This is constructed in order to easily map what value should be placed
        in every column containing NANs within an AGGREGATED dataframe. Saves results to a csv file (since it takes a while to calculate).
        
    Input:
        icu_type_training_data [pd.DataFrame]: the raw dataframe containing original columns of data
        icu_type [string]: specifies to use either picu training set or cticu training set
    Output:
        NONE
    '''
    # (probaly easier way to do it, wanted to visualize operation over all patients)
    avg_per_patient_picu = icu_type_training_data.groupby(level = 0).mean().mean()
    med_per_patient_picu = icu_type_training_data.groupby(level = 0).median().mean()
    std_per_patient_picu = icu_type_training_data.groupby(level = 0).std().mean()
    max_per_patient_picu = icu_type_training_data.groupby(level = 0).max().mean()
    min_per_patient_picu = icu_type_training_data.groupby(level = 0).min().mean()
    
    # FIX ---> WIP
    cnt_per_patient_picu = icu_type_training_data.groupby(level = 0).apply(lambda x: average_count(x)).mean()

    # construct matrix of values to parse over
    summary_mat = pd.DataFrame({'Average': avg_per_patient_picu,
                                'Median': med_per_patient_picu, 
                                'Std': std_per_patient_picu,
                                'Max': max_per_patient_picu,
                                'Min': min_per_patient_picu, 
                                'Count': cnt_per_patient_picu}, index = icu_type_training_data.columns)
    
    # return the resulting summary matrix
    return summary_mat

# ------------------------------------------------------------------------------------------------------------------------------------------# 
# ------------------------------------------------------------------------------------------------------------------------------------------#     
def fill_all_na(col_to_fill, df_to_fill_na, value_matrix_for_fill):
    '''
    Description:
        Uses a pre-calculated values dataframe to map average metrics into an aggregated dataframe containing empty values.
    Input:
        col_to_fill [pd.Series]: The column containing NANS. This is the arguement x being iterated when using pd.apply
        df_to_fill_na [pd.DataFrame]: The Dataframe containing the original columns with NANs 
        value_matrix_for_fill [pd.DataFrame]: the precalculated population means of a given traing set with (x_i = feature_name_i, y_j = aggregation_j)
    Output:
        no_NA_col [pd.Series]: The original column containing NANS, now filled with the population mean from its corresponding training set
    '''

    # seperate the column to fill into coordinates in the value matrix
    split_col_name = col_to_fill.name.split('_')
    if 'last_value' in col_to_fill.name:
        agg_curr = 'Average'
        col_curr = col_to_fill.name.split('_last_value')[0]
        
    else:
        agg_curr = split_col_name[-1]
        col_curr = col_to_fill.name.split('_' + agg_curr)[0]

    # Find value for filling the column containing NaNs and fill empty values with it
    # Check if aggregation was COUNT since count = 0 implies it was 'empty', hence replace 0's,  not NaN's
    if agg_curr == 'Count':
        fill_value = value_matrix_for_fill.loc[col_curr, agg_curr]
        df_to_fill_na[col_to_fill.name] = col_to_fill.replace(0.0, fill_value)
    else:
        fill_value = value_matrix_for_fill.loc[col_curr, agg_curr]
        df_to_fill_na[col_to_fill.name] = col_to_fill.fillna(fill_value)
    
    # set the new column as a variable for organizational purposes
    no_NA_col = df_to_fill_na[col_to_fill.name]
    
    # return the now filled column
    return no_NA_col