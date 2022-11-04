## currently running this on moose with:
#docker run --rm --gpus all --privileged \
#-v '/mnt/ris/ActFastData/Epic_TS_Prototyping/:/storage1/christopherking/Active/ActFastData/Epic_TS_Prototyping/' \
#-v '/home/christopherking/gitdirs/Epic-Time-Series-codes/:/codes/' \
#-v '/mnt/ris/sandhyat/Output-from-docker-TS/:/output/' \
#cryanking/pytorch-for-ts:0.3 python /codes/testing-Preops+Meds+Flowsheet_model-for-docker.py --nameinfo="testingargs" --outputcsv="test.csv" --preopsDepth=5 --preopsWidthFinal=20

# updated docker command
# docker run --rm --gpus all --privileged \
# -v '/mnt/ris/ActFastData/Epic_TS_Prototyping/:/storage1/christopherking/Active/ActFastData/Epic_TS_Prototyping/' \
# -v '/home/trips/Epic-Time-SeriesModels/Epic-Time-Series-codes/:/codes/' \
# -v '/mnt/ris/sandhyat/Output-from-docker-TS/:/output/' \
# cryanking/pytorch-for-ts:0.3 python /codes/testing-Preops+Meds+Flowsheet_model-for-docker.py --nameinfo="testingargs_larger_targetDis" --outputcsv="test.csv" --preopsDepth=5 --preopsWidthFinal=20 --epochs=10

# docker run --rm --gpus all --privileged \
# -v '/mnt/ris/ActFastData/Epic_TS_Prototyping/:/storage1/christopherking/Active/ActFastData/Epic_TS_Prototyping/' \
# -v '/home/trips/Epic-Time-SeriesModels/Epic-Time-Series-codes/:/codes/' \
# -v '/mnt/ris/sandhyat/Output-from-docker-TS/:/output/' \
# cryanking/pytorch-for-ts:0.3 python /codes/testing-Preops+Meds+Flowsheet_model-for-docker.py --nameinfo="testingargs_larger_targetDis_PreopInt" --outputcsv="test.csv" --preopsDepth=5 --preopsWidthFinal=20 --epochs=10 --preopInitLstmFlow=True --preopInitLstmMed=True

# docker run --rm --gpus all --privileged \
# -v '/mnt/ris/ActFastData/Epic_TS_Prototyping/:/storage1/christopherking/Active/ActFastData/Epic_TS_Prototyping/' \
# -v '/home/trips/Epic-Time-SeriesModels/Epic-Time-Series-codes/:/codes/' \
# -v '/mnt/ris/sandhyat/Output-from-docker-TS/:/output/' \
# cryanking/pytorch-for-ts:0.3 python /codes/testing-Preops+Meds+Flowsheet_model-for-docker.py --nameinfo="testingargs_larger_targetDis_PreopInt_biMed" --outputcsv="test.csv" --preopsDepth=5 --preopsWidthFinal=20 --epochs=10 --preopInitLstmFlow --preopInitLstmMed --BilstmMed

# docker run --rm --gpus all --privileged \
# -v '/mnt/ris/ActFastData/Epic_TS_Prototyping/:/storage1/christopherking/Active/ActFastData/Epic_TS_Prototyping/' \
# -v '/home/trips/Epic-Time-SeriesModels/Epic-Time-Series-codes/:/codes/' \
# -v '/mnt/ris/sandhyat/Output-from-docker-TS/:/output/' \
# cryanking/pytorch-for-ts:0.3 python /codes/testing-Preops+Meds+Flowsheet_model-for-docker.py --nameinfo="ICU_larger_targetDis_PreopInt" --outputcsv="test.csv" --preopsDepth=5 --preopsWidthFinal=20 --epochs=10 --preopInitLstmFlow --preopInitLstmMed --task='icu'

# docker run --rm --gpus all --privileged \
# -v '/mnt/ris/ActFastData/Epic_TS_Prototyping/:/storage1/christopherking/Active/ActFastData/Epic_TS_Prototyping/' \
# -v '/home/trips/Epic-Time-SeriesModels/Epic-Time-Series-codes/:/codes/' \
# -v '/mnt/ris/sandhyat/Output-from-docker-TS/:/output/' \
# cryanking/pytorch-for-ts:0.3 python /codes/testing-Preops+Meds+Flowsheet_model-for-docker.py --nameinfo="0paddingEmbed_larger_targetDis_PreopInt" --outputcsv="test.csv" --preopsDepth=5 --preopsWidthFinal=20 --epochs=10 --preopInitLstmFlow --preopInitLstmMed

# docker run --rm --gpus all --privileged \
# -v '/mnt/ris/ActFastData/Epic_TS_Prototyping/:/storage1/christopherking/Active/ActFastData/Epic_TS_Prototyping/' \
# -v '/home/trips/Epic-Time-SeriesModels/Epic-Time-Series-codes/:/codes/' \
# -v '/mnt/ris/sandhyat/Output-from-docker-TS/:/output/' \
# cryanking/pytorch-for-ts:0.3 python /codes/testing-Preops+Meds+Flowsheet_model-for-docker.py --nameinfo="0paddingEmbed_larger_targetDis_PreopInt_BOWsep" --outputcsv="test.csv" --preopsDepth=5 --preopsWidthFinal=20 --epochs=10 --preopInitLstmFlow --preopInitLstmMed

# docker run --rm --gpus all --privileged \
# -v '/mnt/ris/ActFastData/Epic_TS_Prototyping/:/storage1/christopherking/Active/ActFastData/Epic_TS_Prototyping/' \
# -v '/home/trips/Epic-Time-SeriesModels/Epic-Time-Series-codes/:/codes/' \
# -v '/mnt/ris/sandhyat/Output-from-docker-TS/:/output/' \
# cryanking/pytorch-for-ts:0.3 python /codes/testing-Preops+Meds+Flowsheet_model-for-docker.py --nameinfo="MedWordsEmb_larger_targetDis_PreopInt" --outputcsv="test.csv" --preopsDepth=5 --preopsWidthFinal=20 --epochs=10 --preopInitLstmFlow --preopInitLstmMed

import json
import os
import sys, argparse
import numpy as np
import pandas as pd

from pyarrow import feather  # directly writing import pyarrow didn't work
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, \
    RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix
from datetime import datetime

import Preops_processing as pps
import preop_flow_med_bow_model 
from sklearn.metrics import accuracy_score
# TODO: pick better defaults
# TODO: add L1+L2 by directly accessing weights and add to loss instead of applying the same weight decay to everything in the optimized
# TODO: make the same modifications to using the word-sequence
# TODO: MLP for final state (currently fixed at 2 layer)

parser = argparse.ArgumentParser(description='HP for ML optimization')

## for the preops before concat to ts output
parser.add_argument("--preopsDepth",  default=3, type=int) #
parser.add_argument("--preopsWidth",  default=10, type=int) #
parser.add_argument("--preopsWidthFinal",  default=5, type=int) #
parser.add_argument("--preopsL2",  default=0.2, type=float)
parser.add_argument("--preopsL1",  default=0.1, type=float)
parser.add_argument("--preopsBN", default=False, action='store_true') ## not implemented

## for the bow before concat to ts output
parser.add_argument("--bowDepth",  default=3, type=int) #
parser.add_argument("--bowWidth",  default=300, type=int) #
parser.add_argument("--bowWidthFinal",  default=10, type=int) #
parser.add_argument("--bowL2",  default=0.2, type=float)
parser.add_argument("--bowL1",  default=0.1, type=float)

## for processing medication IDs (or the post-embedding words)
parser.add_argument("--lstmMedEmbDim",  default=5, type=int) #
parser.add_argument("--lstmMedDepth",  default=1, type=int)     #
parser.add_argument("--lstmMedWidth",  default=5, type=int) #
parser.add_argument("--lstmMedL2",  default =0.2 , type=float)
parser.add_argument("--lstmMedDrop",  default=0., type=float)  #
parser.add_argument("--preopInitLstmMed", default=True, action='store_true')
parser.add_argument("--BilstmMed", default=False, action='store_true')


## for processing words within a medication name
parser.add_argument("--lstmWordEmbDim",  default=5, type=int) # uses lstmMedEmbDim
parser.add_argument("--lstmWordDepth",  default=1, type=int)                 #
parser.add_argument("--lstmWordWidth",  default=5, type=int) ## not implemented
parser.add_argument("--lstmWordL2",  default =0. , type=float) ## not implemented
parser.add_argument("--lstmWordDrop",  default=0., type=float)  ## not implemented               

## generic dropout of med entry data
parser.add_argument("--lstmRowDrop",  default=0., type=float)    #              

## for processing medication units
## TODO: there is not proper support for units embed dim != med embeed dim or 1, you would have to add a fc layer to make the arrays conformable (or switch to concatenate instead of multiply)
parser.add_argument("--lstmUnitExpand", default=False, action='store_true') #

## for processing flowsheet data
## It's not clear that having 2 LSTMs is good or necessary instead of concatenate the inputs at each timestep
parser.add_argument("--lstmFlowDepth",  default=1, type=int)  #
parser.add_argument("--lstmFlowWidth", default=5 , type=int) #
parser.add_argument("--lstmFlowL2",  default=0.2, type=float)
parser.add_argument("--lstmFlowDrop",  default=0., type=float)
parser.add_argument("--preopInitLstmFlow", default=True, action='store_true')
parser.add_argument("--BilstmFlow", default=False, action='store_true')

## for the MLP combining preop and LSTM outputs
parser.add_argument("--finalDrop",  default=.4, type=float)  #               
parser.add_argument("--finalWidth",  default=10, type=int)   #              
parser.add_argument("--finalDepth",  default=3, type=int)    ## not implemented, fixed at   preopsWidthFinal           
parser.add_argument("--finalBN", default=False, action='store_true') #

## learning parametersq
parser.add_argument("--batchSize",  default=32, type=int) #
parser.add_argument("--learningRate",  default=1e-3, type=float) #
parser.add_argument("--learningRateFactor",  default=0.1, type=float) #
parser.add_argument("--LRPatience",  default=2, type=int) #
parser.add_argument("--epochs",  default=5, type=int) #
parser.add_argument("--XavOrthWeightInt", default=False, action='store_true')  # changes torch's weight initialization to xavier and orthogonal


## task and setup parameters
parser.add_argument("--task",  default="endofcase") #
parser.add_argument("--binaryEOC", default=True, action='store_true')  #
parser.add_argument("--drugNamesNo", default=True,  action='store_true') #
parser.add_argument("--skipPreops",  default=False, action='store_true') # True value of this will only use bow
parser.add_argument("--sepPreopsBow",  default=True, action='store_true') # True value of this variable would treat bow and preops as sep input and have different mlps for them. Also, both skipPreops and sepPreopsBow can't be True at the same time
parser.add_argument("--trainTime", default=True, action='store_true')
parser.add_argument("--testcondition", default='None') # options in  {None, preopOnly, MedOnly, FlowOnly, MedFlow }
parser.add_argument("--randomSeed", default=100, type=int )



## output parameters
parser.add_argument("--git",  default="") # intended to be $(git --git-dir ~/target_dir/.git rev-parse --verify HEAD)
parser.add_argument("--nameinfo",  default="") #
parser.add_argument("--outputcsv",  default="") #

## to add: row-wise drop of TS
##         person-wise drop of preop (less procedure)
##         rate of fc layer contraction
##         output location
##         git state
##         name info for saving
##         hash model state - save a unique name (avoid getting overwritten)

args = parser.parse_args()
if __name__ == "__main__":
  globals().update(args.__dict__) ## it would be better to change all the references to args.thing


# reproducibility settings
# random_seed = 1 # or any of your favorite number
torch.manual_seed(randomSeed)
torch.cuda.manual_seed(randomSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(randomSeed)


# reading the preop and outcome feather files
preops = feather.read_feather('/storage1/christopherking/Active/ActFastData/Epic_TS_Prototyping/preops.feather')
#Post_op_los = feather.read_feather('/storage1/christopherking/Active/ActFastData/Epic_TS_Prototyping/los_out.feather')
# outcomes = feather.read_feather('/storage1/christopherking/Active/ActFastData/Epic_TS_Prototyping/outcomes.feather')

outcomes = pd.read_csv('/storage1/christopherking/Active/ActFastData/Epic_TS_Prototyping/epic_outcomes.csv')
epic_orlogids = pd.read_csv('/storage1/christopherking/Active/ActFastData/Epic_TS_Prototyping/epic_orlogid_codes.csv')
person_to_orlogid = feather.read_feather('/storage1/christopherking/Active/ActFastData/Epic_TS_Prototyping/patient_id_map.feather')

# getting the person_integer ffor the outcomes large file
outcomes.rename(columns={'orlogid':'orlogid_encoded'}, inplace=True)
outcomes = outcomes.join(epic_orlogids.set_index('orlogid_encoded'), on='orlogid_encoded')
outcomes = outcomes.join(person_to_orlogid.set_index('orlogid'), on='orlogid')
outcomes.drop(columns=epic_orlogids.columns, inplace=True)


end_of_case_times = feather.read_feather('/storage1/christopherking/Active/ActFastData/Epic_TS_Prototyping/end_of_case_times.feather')

binary_outcome = task not in ['postop_pred', 'pod3_hct']

# exclude very short cases (this also excludes some invalid negative times)
end_of_case_times = end_of_case_times.loc[end_of_case_times['endtime'] > 30]

if task == 'endofcase':
    # updating the end_of_case_times targets for bigger distribution;
    """ DONT FORGET TO change the label threshold to 25 also in the masking transform function """
    end_of_case_times = end_of_case_times.loc[end_of_case_times['endtime'] > 60] ## cases that are too short
    end_of_case_times = end_of_case_times.loc[end_of_case_times['endtime'] < 25+511] ## cases that are too long
    end_of_case_times['true_test'] = end_of_case_times['endtime'] - 10
    end_of_case_times['t1'] = end_of_case_times['true_test'] -30
    end_of_case_times['t2'] = end_of_case_times['true_test'] -35 # temporary just to make sure nothing breaks; not being used
    end_of_case_times['t3'] = end_of_case_times['true_test'] -40 # temporary just to make sure nothing breaks; not being used
    ## TODO: do something with very long cases
else :
    end_of_case_times['endtime'] = np.minimum(end_of_case_times['endtime'] , 511)
# breakpoint()
# outcome
icu_outcome = outcomes[['person_integer','ICU']]
icu_outcome.loc[icu_outcome ['ICU'] == True, 'ICU'] = 1 
icu_outcome.loc[icu_outcome ['ICU'] == False, 'ICU'] = 0
icu_outcome['ICU']=icu_outcome['ICU'].astype(int)

mortality_outcome = outcomes[['person_integer', 'death_in_30']]
mortality_outcome.loc[mortality_outcome ['death_in_30'] == True, 'death_in_30'] = 1
mortality_outcome.loc[mortality_outcome ['death_in_30'] == False, 'death_in_30'] = 0
mortality_outcome['death_in_30']=mortality_outcome['death_in_30'].astype(int)

aki_outcome = outcomes[['person_integer', 'post_aki_status']]
if task == 'aki1':
    aki_outcome.loc[aki_outcome['post_aki_status'] >= 1, 'post_aki_status'] = 1
    aki_outcome.loc[aki_outcome['post_aki_status'] < 1, 'post_aki_status'] = 0
if task == 'aki2':
    aki_outcome.loc[aki_outcome['post_aki_status'] < 2, 'post_aki_status'] = 0  # the order matters here otherwise everything will bbecome zero :(; there is aone liner too that can be used
    aki_outcome.loc[aki_outcome['post_aki_status'] >= 2, 'post_aki_status'] = 1
if task == 'aki3':
    aki_outcome.loc[aki_outcome['post_aki_status'] < 3, 'post_aki_status'] = 0  # the order matters here otherwise everything will bbecome zero :(; there is aone liner too that can be used
    aki_outcome.loc[aki_outcome['post_aki_status'] == 3, 'post_aki_status'] = 1
aki_outcome['post_aki_status']=aki_outcome['post_aki_status'].astype(int)


dvt_pe_outcome = outcomes[['person_integer','DVT_PE']]

# flowsheet data
very_dense_flow = feather.read_feather("/storage1/christopherking/Active/ActFastData/Epic_TS_Prototyping/Imputed_very_dense_flow.feather")
very_dense_flow.drop(very_dense_flow[very_dense_flow['timepoint'] > 511].index, inplace = True)
very_dense_flow = very_dense_flow.merge(end_of_case_times[['person_integer','endtime']], on="person_integer")
very_dense_flow = very_dense_flow.loc[very_dense_flow['endtime'] > very_dense_flow['timepoint'] ]
very_dense_flow.drop(["endtime"], axis=1, inplace=True)


other_intra_flow_wlabs = feather.read_feather("/storage1/christopherking/Active/ActFastData/Epic_TS_Prototyping/Imputed_other_flow.feather")
other_intra_flow_wlabs.drop(other_intra_flow_wlabs[other_intra_flow_wlabs['timepoint'] > 511].index, inplace = True)
other_intra_flow_wlabs = other_intra_flow_wlabs.merge(end_of_case_times[['person_integer','endtime']], on="person_integer")
other_intra_flow_wlabs = other_intra_flow_wlabs.loc[other_intra_flow_wlabs['endtime'] > other_intra_flow_wlabs['timepoint'] ]
other_intra_flow_wlabs.drop(["endtime"], axis=1, inplace=True)

# reading the med files
all_med_data = feather.read_feather('/storage1/christopherking/Active/ActFastData/Epic_TS_Prototyping/intraop_meds_filterd.feather')
all_med_data.drop(all_med_data[all_med_data['time'] > 511].index, inplace = True)
all_med_data = all_med_data.merge(end_of_case_times[['person_integer','endtime']], on="person_integer")
all_med_data = all_med_data.loc[all_med_data['endtime'] > all_med_data['time'] ]
all_med_data.drop(["endtime"], axis=1, inplace=True)

drug_med_ids = all_med_data[['person_integer', 'time', 'drug_position', 'med_integer']]

if drugNamesNo == True:
    drug_med_id_map = feather.read_feather('/storage1/christopherking/Active/ActFastData/Epic_TS_Prototyping/med_id_map.feather')
    drug_words = None
    word_id_map = None
else:
    drug_words = feather.read_feather('/storage1/christopherking/Active/ActFastData/Epic_TS_Prototyping/drug_words.feather')
    drug_words.drop(drug_words[drug_words['timepoint'] > 511].index, inplace = True)
    word_id_map = feather.read_feather('/storage1/christopherking/Active/ActFastData/Epic_TS_Prototyping/word_id_map.feather')
    drug_med_id_map = None




drug_dose = all_med_data[['person_integer', 'time', 'drug_position', 'unit_integer', 'dose']]

unit_id_map = feather.read_feather('/storage1/christopherking/Active/ActFastData/Epic_TS_Prototyping/unit_id_map.feather')

vocab_len_units = len(unit_id_map)

if drugNamesNo == False:
    vocab_len_words = len(word_id_map)
else:
    vocab_len_med_ids = len(drug_med_id_map)


binary_outcome_list = ['UTI', 'CVA', 'PNA', 'PE', 'DVT', 'AF', 'arrest', 'VTE', 'GI', 'SSI', 'pulm', 'cardiac', 'postop_trop_crit', 'postop_trop_high']

if task == 'postop_pred':
    outcome_df = outcomes[['person_integer', 'postop_los']]
elif task == 'pod3_hct':
    outcome_df = outcomes[['person_integer', 'pod3_hct']]
elif task in binary_outcome_list:
    if task == 'VTE':
        temp_outcome = outcomes[['person_integer']]
        temp_outcome[task] = np.where(outcomes['DVT'] == True, 1, 0) + np.where(outcomes['PE'] == True, 1, 0)
        temp_outcome.loc[temp_outcome[task] == 2, task] = 1
    else:
        temp_outcome = outcomes[['person_integer', task]]
        temp_outcome.loc[temp_outcome[task] == True, task] = 1
        temp_outcome.loc[temp_outcome[task] == False, task] = 0
    temp_outcome[task] = temp_outcome[task].astype(int)
    outcome_df = temp_outcome
elif (task == 'dvt_pe'):
    outcome_df = dvt_pe_outcome
elif (task == 'icu'):
    outcome_df = icu_outcome
elif (task == 'mortality'):
    outcome_df = mortality_outcome
elif (task == 'aki1' or task == 'aki2' or task == 'aki3'):
    outcome_df = aki_outcome
elif (task == 'endofcase'):
    outcome_df = end_of_case_times[['person_integer', 'true_test']]
else:
    raise Exception("outcome not handled")



## intersect 3 mandatory data sources: preop, outcome, case end times
combined_case_set = list(set(outcome_df["person_integer"].values ).intersection( 
  set(end_of_case_times['person_integer'].values ) ).intersection(
  set(preops['person_integer'].values) ) )

# combined_case_set = np.random.choice(combined_case_set, 8000, replace=False)  
outcome_df = outcome_df.loc[outcome_df['person_integer'].isin(combined_case_set) ]
preops = preops.loc[preops['person_integer'].isin(combined_case_set) ]
end_of_case_times = end_of_case_times.loc[end_of_case_times['person_integer'].isin(combined_case_set) ]


outcome_df.set_axis(["person_integer","outcome"], axis=1, inplace=True)
  
# checking for NA and other filters
outcome_df = outcome_df.loc[outcome_df['person_integer'].isin(preops["person_integer"].unique() ) ]
outcome_df = outcome_df.dropna(axis=0).sort_values(["person_integer"]).reset_index(drop=True)
## reindex person_integer
new_index = outcome_df["person_integer"].copy().reset_index().rename({"index":"new_person"}, axis=1)


## drop missing data
drug_dose= drug_dose.merge(new_index, on="person_integer", how="inner").drop(["person_integer"], axis=1).rename({"new_person":"person_integer"}, axis=1)
preops= preops.merge(new_index, on="person_integer", how="inner").drop(["person_integer"], axis=1).rename({"new_person":"person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)
endtimes= end_of_case_times.merge(new_index, on="person_integer", how="inner").drop(["person_integer"], axis=1).rename({"new_person":"person_integer"}, axis=1).sort_values(["person_integer"]).reset_index(drop=True)
very_dense_flow= very_dense_flow.merge(new_index, on="person_integer", how="inner").drop(["person_integer"], axis=1).rename({"new_person":"person_integer"}, axis=1)
other_intra_flow_wlabs= other_intra_flow_wlabs.merge(new_index, on="person_integer", how="inner").drop(["person_integer"], axis=1).rename({"new_person":"person_integer"}, axis=1)
if drug_words is not None:
  drug_words= drug_words.merge(new_index, on="person_integer", how="inner").drop(["person_integer"], axis=1).rename({"new_person":"person_integer"}, axis=1)

if drug_med_ids is not None:
  drug_med_ids= drug_med_ids.merge(new_index, on="person_integer", how="inner").drop(["person_integer"], axis=1).rename({"new_person":"person_integer"}, axis=1)


outcome_df.drop(["person_integer"], axis=1, inplace=True)
outcome_df.reset_index(inplace=True)
outcome_df.rename({"index":"person_integer"}, axis=1,inplace=True)


bow_cols = [col for col in preops.columns if 'bow' in col]
bow_input = preops[bow_cols].copy()
preops['BOW_NA'] = np.where(np.isnan(preops[bow_cols[0]]), 1, 0)
bow_input.fillna(0, inplace=True)
preops = preops.drop(columns=bow_cols)


 ## I suppose these could have sorted differently
 ## TODO apparently, torch.from_numpy shares the memory buffer and inherits type 
index_med_ids = torch.tensor(drug_med_ids[['person_integer', 'time', 'drug_position']].values, dtype=int)
index_med_dose= torch.tensor(drug_dose[['person_integer', 'time', 'drug_position']].values, dtype=int)
value_med_dose = torch.tensor(drug_dose['dose'].astype('float').values, dtype=float)
value_med_unit = torch.tensor(drug_dose['unit_integer'].values, dtype=int)

add_unit = 0 in value_med_unit.unique()
dense_med_units = torch.sparse_coo_tensor(torch.transpose(index_med_dose, 0, 1), value_med_unit + add_unit, dtype=torch.int32)
dense_med_dose = torch.sparse_coo_tensor(torch.transpose(index_med_dose, 0,1), value_med_dose, dtype=torch.float32)

if drugNamesNo == True:
    value_med_ids = torch.tensor(drug_med_ids['med_integer'].values, dtype=int)
    add_med = 0 in value_med_ids.unique()
    dense_med_ids = torch.sparse_coo_tensor(torch.transpose(index_med_ids, 0 ,1), value_med_ids+add_med, dtype=torch.int32)
else: ## not considered
    drug_words.dropna(axis=0, inplace=True)
    # convert name and unit+dose data seperately into the required format
    drug_words['time'] = drug_words['time'].astype('int64')
    index_med_names = torch.tensor(drug_words[['person_integer', 'time', 'drug_position', 'word_position']].values,dtype=int)
    value_med_name = torch.tensor(drug_words['word_integer'].values, dtype=int)
    add_name = 0 in value_med_name.unique()
    dense_med_names = torch.sparse_coo_tensor(torch.transpose(index_med_names, 0, 1),
                                              value_med_name + add_name, dtype=torch.int32).to_dense()

""" TS flowsheet proprocessing """

index_med_other_flow = torch.tensor(other_intra_flow_wlabs[['person_integer', 'timepoint', 'measure_index']].values,dtype=int)
value_med_other_flow = torch.tensor(other_intra_flow_wlabs['VALUE'].values)
flowsheet_other_flow = torch.sparse_coo_tensor(torch.transpose(index_med_other_flow, 0, 1),
                                           value_med_other_flow, dtype=torch.float32)

index_med_very_dense = torch.tensor(very_dense_flow[['person_integer', 'timepoint','measure_index']].values, dtype=int)
value_med_very_dense = torch.tensor(very_dense_flow['VALUE'].values)
flowsheet_very_dense = torch.sparse_coo_tensor(torch.transpose(index_med_very_dense, 0, 1),
                                          value_med_very_dense, dtype=torch.float32).to_dense() ## this is memory heavy and could be skipped, only because it is making a copy not really because it is harder to store
flowsheet_very_dense = torch.cumsum(flowsheet_very_dense, dim=1)

# trying to concatenate the two types of flowsheet tensors at the measure_index dimension
#flowsheet_dense_comb = torch.cat((flowsheet_very_dense, flowsheet_other_flow), dim=2)
total_flowsheet_measures = other_intra_flow_wlabs['measure_index'].unique().max()+1 + very_dense_flow['measure_index'].unique().max() +1 # plus 1 because of the python indexing from 0


        
# breakpoint()

## TODO: I haven't reviewed this
if(trainTime):
    # use all the data to make prediction
    preops_tr, preops_val, preops_te, train_index, valid_index, test_index, _ = pps.preprocess_train(preops, skipPreops, y_outcome=outcome_df["outcome"].values , binary_outcome=binary_outcome, valid_size =0.00005)
else :
    # reading metadata file generated during training time
    md_f = open('/output/preops_metadata.json')
    metadata =  json.load(md_f)
    if task == 'postop_pred' or task =='endofcase':
        preops_te = pps.preprocess_inference(preops, skipPreops, metadata)
    else:
        preops_te = pps.preprocess_inference(preops, skipPreops, metadata)



# breakpoint()
# if task == 'endofcase': ##I only included the first two timepoints; doing the rest requires either excluding cases so that all 4 are defined or more complex indexing
#   data_tr = [ torch.vstack([torch.hstack([torch.tensor(preops_tr.to_numpy(), dtype=torch.float32), torch.tensor(endtimes.iloc[train_index]["true_test"].values , dtype=int).reshape(len(preops_tr),1)]),
#                             torch.hstack([torch.tensor(preops_tr.to_numpy(), dtype=torch.float32), torch.tensor(endtimes.iloc[train_index]["t1"].values , dtype=int).reshape(len(preops_tr),1)])] ) , ## appended the current time here; can't do it in preops processing because for the two cases true_test and t1 are seperate
#             torch.hstack([torch.tensor(endtimes.iloc[train_index]["true_test"].values , dtype=int) , torch.tensor(endtimes.iloc[train_index]["t1"].values , dtype=int)  ] ) ,  ## the durations are hstacked because single columns are apparantly transposed before stacking and hence it was becoming a [2X len(data)] creating size inconsistencies
#             torch.vstack([torch.tensor(bow_input.iloc[train_index].to_numpy(), dtype=torch.float32)]*2 ) ,
#             torch.vstack([torch.index_select( dense_med_ids , 0 , torch.tensor(train_index) ).coalesce()]*2 ) ,
#             torch.vstack([torch.index_select( dense_med_dose , 0 , torch.tensor(train_index) ).coalesce()]*2 ) ,
#             torch.vstack([torch.index_select( dense_med_units , 0 , torch.tensor(train_index) ).coalesce()]*2 ) ,
#             torch.vstack([flowsheet_very_dense[train_index,:,:]]*2 ) ,
#             torch.vstack([torch.index_select( flowsheet_other_flow , 0 , torch.tensor(train_index) ).coalesce()]*2 ) ,
#             torch.from_numpy( np.repeat([1,0], len(train_index) ))
#             ]
#   # breakpoint()
#   data_te = [ torch.vstack([torch.hstack([torch.tensor(preops_te.to_numpy(), dtype=torch.float32), torch.tensor(endtimes.iloc[test_index]["true_test"].values , dtype=int).reshape(len(preops_te),1)]),
#                             torch.hstack([torch.tensor(preops_te.to_numpy(), dtype=torch.float32), torch.tensor(endtimes.iloc[test_index]["t1"].values , dtype=int).reshape(len(preops_te),1)])] ) ,
#             torch.hstack([torch.tensor(endtimes.iloc[test_index]["true_test"].values , dtype=int) , torch.tensor(endtimes.iloc[test_index]["t1"].values , dtype=int)  ] ) ,
#             torch.vstack([torch.tensor(bow_input.iloc[test_index].to_numpy(), dtype=torch.float32)]*2 ) ,
#             torch.vstack([torch.index_select( dense_med_ids , 0 , torch.tensor(test_index) ).coalesce()]*2 ) ,
#             torch.vstack([torch.index_select( dense_med_dose , 0 , torch.tensor(test_index) ).coalesce()]*2 ) ,
#             torch.vstack([torch.index_select( dense_med_units , 0 , torch.tensor(test_index) ).coalesce()]*2 ) ,
#             torch.vstack([flowsheet_very_dense[test_index,:,:]]*2 ) ,
#             torch.vstack([torch.index_select( flowsheet_other_flow , 0 , torch.tensor(test_index) ).coalesce()]*2 ) ,
#             torch.from_numpy( np.repeat([1,0], len(test_index) ))
#             ]
#   data_va = [  torch.vstack([torch.hstack([torch.tensor(preops_val.to_numpy(), dtype=torch.float32), torch.tensor(endtimes.iloc[valid_index]["true_test"].values , dtype=int).reshape(len(preops_val),1)]),
#                             torch.hstack([torch.tensor(preops_val.to_numpy(), dtype=torch.float32), torch.tensor(endtimes.iloc[valid_index]["t1"].values , dtype=int).reshape(len(preops_val),1)])] ) ,
#             torch.hstack([torch.tensor(endtimes.iloc[valid_index]["true_test"].values, dtype=int ) , torch.tensor(endtimes.iloc[valid_index]["t1"].values, dtype=int )  ] ) ,
#             torch.vstack([torch.tensor(bow_input.iloc[valid_index].to_numpy(), dtype=torch.float32) ]*2 ) ,
#             torch.vstack([torch.index_select( dense_med_ids , 0 , torch.tensor(valid_index) ).coalesce()]*2 ) ,
#             torch.vstack([torch.index_select( dense_med_dose , 0 , torch.tensor(valid_index) ).coalesce()]*2 ) ,
#             torch.vstack([torch.index_select( dense_med_units , 0 , torch.tensor(valid_index) ).coalesce()]*2 ) ,
#             torch.vstack([flowsheet_very_dense[valid_index,:,:]]*2 ) ,
#             torch.vstack([torch.index_select( flowsheet_other_flow , 0 , torch.tensor(valid_index) ).coalesce()]*2 ) ,
#             torch.from_numpy( np.repeat([1,0], len(valid_index) ))
#             ]
# else :
#   data_tr = [
#             torch.tensor(preops_tr.to_numpy(), dtype=torch.float32),
#             torch.tensor(endtimes.iloc[train_index]["endtime"].values , dtype=int) ,
#             torch.tensor(bow_input.iloc[train_index].to_numpy(), dtype=torch.float32)  ,
#             torch.index_select( dense_med_ids , 0 , torch.tensor(train_index) ).coalesce() , 
#             torch.index_select( dense_med_dose , 0 , torch.tensor(train_index) ).coalesce() , 
#             torch.index_select( dense_med_units , 0 , torch.tensor(train_index) ).coalesce() , 
#             flowsheet_very_dense[train_index,:,:] , 
#             torch.index_select( flowsheet_other_flow , 0 , torch.tensor(train_index) ).coalesce() ,
#             torch.tensor(outcome_df.iloc[train_index]["outcome"].values)
#             ]
#   data_te = [
#             torch.tensor(preops_te.to_numpy(), dtype=torch.float32),
#             torch.tensor(endtimes.iloc[test_index]["endtime"].values , dtype=int) ,
#             torch.tensor(bow_input.iloc[test_index].to_numpy(), dtype=torch.float32)  ,
#             torch.index_select( dense_med_ids , 0 , torch.tensor(test_index) ).coalesce() , 
#             torch.index_select( dense_med_dose , 0 , torch.tensor(test_index) ).coalesce() , 
#             torch.index_select( dense_med_units , 0 , torch.tensor(test_index) ).coalesce() , 
#             flowsheet_very_dense[test_index,:,:] , 
#             torch.index_select( flowsheet_other_flow , 0 , torch.tensor(test_index) ).coalesce() ,
#             torch.tensor(outcome_df.iloc[test_index]["outcome"].values)
#             ]
#   data_va = [
#             torch.tensor(preops_val.to_numpy(), dtype=torch.float32),
#             torch.tensor(endtimes.iloc[valid_index]["endtime"].values , dtype=int) ,
#             torch.tensor(bow_input.iloc[valid_index].to_numpy(), dtype=torch.float32)  ,
#             torch.index_select( dense_med_ids , 0 , torch.tensor(valid_index) ).coalesce() , 
#             torch.index_select( dense_med_dose , 0 , torch.tensor(valid_index) ).coalesce() , 
#             torch.index_select( dense_med_units , 0 , torch.tensor(valid_index) ).coalesce() , 
#             flowsheet_very_dense[valid_index,:,:] , 
#             torch.index_select( flowsheet_other_flow , 0 , torch.tensor(valid_index) ).coalesce() ,
#             torch.tensor(outcome_df.iloc[valid_index]["outcome"].values)
#             ]



if(False):
  train_dataset = TensorDataset(*data_tr)
  train_loader = DataLoader(train_dataset, batchSize, shuffle=True, collate_fn=preop_flow_med_bow_model.collate_time_series)

  test_dataset = TensorDataset(*data_te)
  test_loader = DataLoader(train_dataset, batchSize, shuffle=True, collate_fn=preop_flow_med_bow_model.collate_time_series)

  valid_dataset = TensorDataset(*data_va)
  valid_loader = DataLoader(train_dataset, batchSize, shuffle=False, collate_fn=preop_flow_med_bow_model.collate_time_series)

# def insert(df, i, df_add):
#     # insert data to ceratin rows
#     df1 = df.iloc[:i, :]
#     df2 = df.iloc[i:, :]
#     df_new = pd.concat([df1, df_add, df2], ignore_index=True)
#     return df_new

#get index for time series

predictions_holder = pd.DataFrame({'people': [],
                    'time': [],
                    'true value':[],
                   'Prediction': []})
model_saving_path = '/home/research/chuz/model.pth'

model = torch.load(model_saving_path)
time_index = [i for i in range(outcome_df.shape[0])]
times=0
#result = pd.read_feather("/storage1/christopherking/Active/ActFastData/Epic_TS_Prototyping/People_Prediction_DifferetTime.feather")
for time in range(10,511,10):
    start=datetime.now() 
    indices = torch.LongTensor([i for i in range(0,time+1)])
    dense_med_ids_time = torch.index_select(dense_med_ids,1,indices)
    dense_med_dose_time = torch.index_select(dense_med_dose,1,indices)
    dense_med_units_time = torch.index_select(dense_med_units,1,indices)
    new_endtimes = endtimes["endtime"].copy(deep=True)
    for i in range(endtimes.shape[0]):
        new_endtimes[i] = min(endtimes["endtime"][i],time)
    if(task=="icu"):
        data_tr = [
                    torch.tensor(preops_tr.to_numpy(), dtype=torch.float32),
                    torch.tensor(new_endtimes , dtype=int) ,
                    torch.tensor(bow_input.to_numpy(), dtype=torch.float32)  ,
                    dense_med_ids_time.coalesce() , 
                    dense_med_dose_time.coalesce() , 
                    dense_med_units_time.coalesce() , 
                    flowsheet_very_dense[time_index,:,:] , 
                    flowsheet_other_flow.coalesce() ,
                    torch.tensor(outcome_df["outcome"])
                    ]
    print("current start time:", time)



    device = torch.device('cuda')
    # breakpoint()
    # model = preop_flow_med_bow_model.TS_lstm_Med_index(
    #   v_units=vocab_len_units,
    #   v_med_ids=vocab_len_med_ids,
    #   e_dim_med_ids=lstmMedEmbDim,
    #   e_dim_units=lstmUnitExpand,
    #   preops_init_med=preopInitLstmMed,
    #   preops_init_flow=preopInitLstmFlow,
    #   lstm_hid=lstmMedWidth,
    #   lstm_flow_hid=lstmFlowWidth,
    #   lstm_num_layers=lstmMedDepth,
    #   lstm_flow_num_layers=lstmFlowDepth,
    #   bilstm_med = BilstmMed,
    #   bilstm_flow = BilstmFlow,
    #   linear_out=1,
    #   p_idx_med_ids=0,  # putting these 0 because the to dense sets everything not available as 0
    #   p_idx_units=0,
    #   p_time=lstmMedDrop,
    #   p_flow=lstmFlowDrop,
    #   p_rows=lstmRowDrop,
    #   p_final=finalDrop,
    #   binary= binary_outcome,
    #   hidden_units=preopsWidth,
    #   hidden_units_final=preopsWidthFinal,
    #   hidden_depth=preopsDepth,
    #   finalBN=finalBN,
    #   input_shape=data_tr[0].shape[1], # this is done so that I dont have to write a seperate condition for endofcase where the current time is being appended to preops
    #   hidden_units_bow=bowWidth,
    #   hidden_units_final_bow=bowWidthFinal,
    #   hidden_depth_bow=bowDepth,
    #   input_shape_bow=len(bow_input.columns),
    #   num_flowsheet_feat=total_flowsheet_measures,
    #     weight_decay_preopsL2=preopsL2,
    #     weight_decay_preopsL1=preopsL1,
    #     weight_decay_bowL2=bowL2,
    #     weight_decay_bowL1=bowL1,
    #     weight_decay_LSTMmedL2=lstmMedL2,
    #     weight_decay_LSTMflowL2=lstmFlowL2,
    #     weightInt = XavOrthWeightInt
    #   ).to(device)

    # optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=1e-5)

    # # lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=LRPatience, verbose=True, factor=learningRateFactor)



    #make prediction by the model:
    loss_te = 0
    loss_te_cls = 0
    with torch.no_grad():
        model.eval()
        true_y_test = []
        pred_y_test = []
        nbatch = data_tr[0].shape[0] // batchSize
        for i in range(nbatch):
            these_index = torch.tensor(list(range(i*batchSize,(i+1)*batchSize ) ), dtype=int)
            local_data =[ torch.index_select(x,  0 ,  these_index )  for x in data_tr]
            #need to deal with
            data_valid = preop_flow_med_bow_model.collate_time_series( [ [ x[i]  for x in local_data]  for i in range(local_data[0].shape[0])]  )

            if True:  # temporary for ablation studies
                    ## inputs are assumed to be: [preops, durations, bow, med_index, med_dose, med_unit, flow_dense, flow_sparse (optional), labels (optional)]
                if testcondition == "preopOnly":
                    data_valid[0][3] = torch.zeros(data_valid[0][3].shape)
                    data_valid[0][4] = torch.zeros(data_valid[0][4].shape)
                    data_valid[0][5] = torch.zeros(data_valid[0][5].shape)
                    data_valid[0][6] = torch.zeros(data_valid[0][6].shape)
                if testcondition == 'MedOnly':
                    data_valid[0][0] = torch.zeros(data_valid[0][0].shape)
                    data_valid[0][2] = torch.zeros(data_valid[0][2].shape)
                    data_valid[0][6] = torch.zeros(data_valid[0][6].shape)
                if testcondition == 'FlowOnly':
                    data_valid[0][0] = torch.zeros(data_valid[0][0].shape)
                    data_valid[0][2] = torch.zeros(data_valid[0][2].shape)
                    data_valid[0][3] = torch.zeros(data_valid[0][3].shape)
                    data_valid[0][4] = torch.zeros(data_valid[0][4].shape)
                    data_valid[0][5] = torch.zeros(data_valid[0][5].shape)
                if testcondition == 'MedFlow':
                    data_valid[0][0] = torch.zeros(data_valid[0][0].shape)
                    data_valid[0][2] = torch.zeros(data_valid[0][2].shape)

            for data_index in [0,2,3,4,5,6]:
                data_valid[0][data_index] = data_valid[0][data_index].to(device)
            y_pred, reg_loss = model(*data_valid[0])

                
            true_y_test.append(data_valid[1].float().detach().numpy())
            pred_y_test.append(y_pred.squeeze(-1).cpu().detach().numpy())
    true_y_test= np.concatenate(true_y_test)
    pred_y_test= np.concatenate(pred_y_test)
   
    result = pd.DataFrame({'people': [int(i) for i in range(86400)],
                    'time': [time]*86400,
                    'true value':true_y_test,
                   'Prediction': pred_y_test})
    predictions_holder = predictions_holder.append(result,ignore_index=True)
    # for i in range(86400):
    #     new_row = pd.DataFrame({'people': [i],
    #                 'time': [time],
    #                 'true value':[true_y_test[i]],
    #                 'Prediction': [pred_y_test[i]]})
    #     epoches = (time-10)/10 
    #     if(time>10):
    #         #get the index for the former same people
    #         next_index = int((epoches+1)*i+epoches)
    #         # temp = result[result['people']==i]
    #         # next_index = temp[temp['time']==(time-10)].index[0]+1
    #         result = insert(result,next_index,new_row)
    #     else:
    #         result = result.append(new_row)
    end = datetime.now() 
    print("cost:",(end-start).seconds/60,"minutes")   
predictions_holder.to_feather("/storage1/christopherking/Active/ActFastData/Epic_TS_Prototyping/People_Prediction_DifferetTime.feather")





# initializing the loss function
# if not binary_outcome:
#   criterion = torch.nn.MSELoss()
# else :
#   criterion = torch.nn.BCELoss()

# total_train_loss = []
# total_test_loss = []
# for epoch in range(epochs):
#   loss_tr = 0
#   loss_tr_cls = 0
#   model.train()
# ## the default __getitem__ is like 2 orders of magnitude slower
#   shuffle_index = torch.randperm(n=data_tr[0].shape[0])
#   nbatch = data_tr[0].shape[0] // batchSize
#   for i in range(nbatch):
#       # breakpoint()
#       these_index = shuffle_index[range(i*batchSize,(i+1)*batchSize ) ]
#       ## this collate method is pretty inefficent for this task but works with the generic DataLoader method
#       local_data =[ torch.index_select(x,  0 ,  these_index )  for x in data_tr]
#       data_train = preop_flow_med_bow_model.collate_time_series( [ [ x[i]  for x in local_data]  for i in range(local_data[0].shape[0])]  )
#   #for i, data_train in enumerate(train_loader):

#       if True:  # temporary for ablation studies
#           ## inputs are assumed to be: [preops, durations, bow, med_index, med_dose, med_unit, flow_dense, flow_sparse (optional), labels (optional)]
#           if testcondition == "preopOnly":
#               data_train[0][3] = torch.zeros(data_train[0][3].shape)
#               data_train[0][4] = torch.zeros(data_train[0][4].shape)
#               data_train[0][5] = torch.zeros(data_train[0][5].shape)
#               data_train[0][6] = torch.zeros(data_train[0][6].shape)
#           if testcondition == 'MedOnly':
#               data_train[0][0] = torch.zeros(data_train[0][0].shape)
#               data_train[0][2] = torch.zeros(data_train[0][2].shape)
#               data_train[0][6] = torch.zeros(data_train[0][6].shape)
#           if testcondition == 'FlowOnly':
#               data_train[0][0] = torch.zeros(data_train[0][0].shape)
#               data_train[0][2] = torch.zeros(data_train[0][2].shape)
#               data_train[0][3] = torch.zeros(data_train[0][3].shape)
#               data_train[0][4] = torch.zeros(data_train[0][4].shape)
#               data_train[0][5] = torch.zeros(data_train[0][5].shape)
#           if testcondition == 'MedFlow':
#               data_train[0][0] = torch.zeros(data_train[0][0].shape)
#               data_train[0][2] = torch.zeros(data_train[0][2].shape)

#       for data_index in [0,2,3,4,5,6]:
#         data_train[0][data_index] = data_train[0][data_index].to(device)
#       ## TODO: this hurts me aesthetically; it would be nice to have the collate function have an option for this. the reason is to avoid passing device as an argument to forward pass. Currently I have the model detect what device the parameters are on to determine where to put the initialized LSTM parameters, so it is device agnostic. I will have to look into the function constructor type class with init and call
#       # reset the gradients back to zero as PyTorch accumulates gradients on subsequent backward passes
#       optimizer.zero_grad()
#       if True:
#         y_pred, reg_loss = model(*data_train[0])
#         cls_loss_tr = criterion(y_pred.squeeze(-1), data_train[1].float().to(device)).float()
#         train_loss = cls_loss_tr + reg_loss
#         train_loss.backward()
#         optimizer.step()
#         loss_tr += train_loss.item()
#         loss_tr_cls += cls_loss_tr.item()
#   loss_tr = loss_tr / data_tr[0].shape[0]
#   loss_tr_cls = loss_tr_cls/ data_tr[0].shape[0]

#   loss_te = 0
#   loss_te_cls = 0
#   with torch.no_grad():
#     model.eval()
#     true_y_test = []
#     pred_y_test = []
#     nbatch = data_te[0].shape[0] // batchSize
#     for i in range(nbatch):
#         these_index = torch.tensor(list(range(i*batchSize,(i+1)*batchSize ) ), dtype=int)
#         local_data =[ torch.index_select(x,  0 ,  these_index )  for x in data_te]
#         data_valid = preop_flow_med_bow_model.collate_time_series( [ [ x[i]  for x in local_data]  for i in range(local_data[0].shape[0])]  )

#         if True:  # temporary for ablation studies
#             ## inputs are assumed to be: [preops, durations, bow, med_index, med_dose, med_unit, flow_dense, flow_sparse (optional), labels (optional)]
#             if testcondition == "preopOnly":
#                 data_valid[0][3] = torch.zeros(data_valid[0][3].shape)
#                 data_valid[0][4] = torch.zeros(data_valid[0][4].shape)
#                 data_valid[0][5] = torch.zeros(data_valid[0][5].shape)
#                 data_valid[0][6] = torch.zeros(data_valid[0][6].shape)
#             if testcondition == 'MedOnly':
#                 data_valid[0][0] = torch.zeros(data_valid[0][0].shape)
#                 data_valid[0][2] = torch.zeros(data_valid[0][2].shape)
#                 data_valid[0][6] = torch.zeros(data_valid[0][6].shape)
#             if testcondition == 'FlowOnly':
#                 data_valid[0][0] = torch.zeros(data_valid[0][0].shape)
#                 data_valid[0][2] = torch.zeros(data_valid[0][2].shape)
#                 data_valid[0][3] = torch.zeros(data_valid[0][3].shape)
#                 data_valid[0][4] = torch.zeros(data_valid[0][4].shape)
#                 data_valid[0][5] = torch.zeros(data_valid[0][5].shape)
#             if testcondition == 'MedFlow':
#                 data_valid[0][0] = torch.zeros(data_valid[0][0].shape)
#                 data_valid[0][2] = torch.zeros(data_valid[0][2].shape)

#         for data_index in [0,2,3,4,5,6]:
#           data_valid[0][data_index] = data_valid[0][data_index].to(device)
#         y_pred, reg_loss = model(*data_valid[0])
#         cls_loss_te = criterion(y_pred.squeeze(-1),data_valid[1].float().to(device)).float()
#         test_loss = cls_loss_te + reg_loss
#         loss_te += test_loss.item()
#         loss_te_cls += cls_loss_te.item()

#         if epoch==epochs-1:  # values from the last epoch
#             # using test data only instead of validation data for evaluation currently because the validation will be done on a seperate data
#             true_y_test.append(data_valid[1].float().detach().numpy())
#             pred_y_test.append(y_pred.squeeze(-1).cpu().detach().numpy())

#     loss_te = loss_te / data_te[0].shape[0]
#     loss_te_cls = loss_te_cls/ data_te[0].shape[0]
#     # display the epoch training and test loss
#     print("epoch : {}/{}, training loss = {:.8f}, validation loss = {:.8f}".format(epoch + 1, epochs, loss_tr_cls,loss_te_cls) )
#     total_train_loss.append(loss_tr)
#     total_test_loss.append(loss_te)

#   scheduler.step(loss_te_cls)

# torch.save(model, model_saving_path)
# model.eval()

# if False:
#     with torch.no_grad():
#       loss_te = 0
#       true_y_test = []
#       pred_y_test =[]
#       ## this calls the entire dataset in the first batch
#       #for i, data_test in enumerate(test_loader):
#       nbatch = data_va[0].shape[0] // batchSize
#       for i in range(nbatch):
#           these_index = torch.tensor(list(range(i*batchSize,(i+1)*batchSize ) ), dtype=int)
#           local_data =[ torch.index_select(x,  0 ,  these_index )  for x in data_va]
#           data_valid = preop_flow_med_bow_model.collate_time_series( [ [ x[i]  for x in local_data]  for i in range(local_data[0].shape[0])]  )
#           for data_index in [0,2,3,4,5,6]:
#             data_valid[0][data_index] = data_valid[0][data_index].to(device)
#           true_y_test.append( data_valid[1].float().detach().numpy() )
#           y_pred, _ = model(*data_valid[0])
#           pred_y_test.append(y_pred.squeeze(-1).cpu().detach().numpy() )

# true_y_test= np.concatenate(true_y_test)
# pred_y_test= np.concatenate(pred_y_test)
# if not binary_outcome:
#     corr_value = np.round(np.corrcoef(np.array(true_y_test), np.array(pred_y_test))[1, 0], 3)
#     print(str(task) + " prediction with correlation ", corr_value)
#     r2value = r2_score(np.array(true_y_test), np.array(pred_y_test))
#     temp_df = pd.DataFrame(columns=['true_value', 'pred_value'])
#     temp_df['true_value'] = np.array(true_y_test)
#     temp_df['pred_value'] = np.array(pred_y_test)
#     temp_df['abs_diff'] = abs(temp_df['true_value'] - temp_df['pred_value'])
#     mae_full = np.round(temp_df['abs_diff'].mean(), 3)
#     print("MAE on the test set ", mae_full)
#     q25, q7, q9 = temp_df['true_value'].quantile([0.25, 0.7, 0.9])

#     firstP_data = temp_df.query('true_value<={high}'.format(high=q25))
#     secondP_data = temp_df.query('{low}<true_value<={high}'.format(low=q25, high=q7))
#     thirdP_data = temp_df.query('{low}<true_value<={high}'.format(low=q7, high=q9))
#     fourthP_data = temp_df.query('{low}<true_value'.format(low=q9))

#     mae_dict = {'<' + str(np.round(q25, decimals=1)): firstP_data['abs_diff'].mean(),
#                 str(np.round(q25, decimals=1)) + "<" + str(np.round(q7, decimals=1)): secondP_data['abs_diff'].mean(),
#                 str(np.round(q7, decimals=1)) + "<" + str(np.round(q9, decimals=1)): thirdP_data['abs_diff'].mean(),
#                 str(np.round(q9, decimals=1)) + "<": fourthP_data['abs_diff'].mean()}

#     stratifying_point_dict = {'<' + str(np.round(q25, decimals=1)): '<' + str(np.round(q25, decimals=1)),
#                               str(np.round(q25, decimals=1)) + "<" + str(np.round(q7, decimals=1)): str(
#                                   np.round(q25, decimals=1)) + "<" + str(np.round(q7, decimals=1)),
#                               str(np.round(q7, decimals=1)) + "<" + str(np.round(q9, decimals=1)): str(
#                                   np.round(q7, decimals=1)) + "<" + str(np.round(q9, decimals=1)),
#                               str(np.round(q9, decimals=1)) + "<": str(np.round(q9, decimals=1)) + "<"}

#     csvdata = {
#         'hp': json.dumps(vars(args)),
#         'Initial_seed': randomSeed,  # this is being done so its easier to differentiate each line in the final csv file
#         'corr': corr_value,
#         'R2': r2value,
#         'MAE': mae_full,
#         'Stratifying_points': stratifying_point_dict,
#         'Stratified_MAE': mae_dict,
#         'git': args.git,
#         'name': args.nameinfo,
#         'target': args.task,
#         'evaltime': datetime.now().strftime("%y-%m-%d-%H:%M:%S")
#     }

#     csvdata = pd.DataFrame(csvdata)
#     outputcsv = os.path.join('/output/', args.outputcsv)
#     if (os.path.exists(outputcsv)):
#         csvdata.to_csv(outputcsv, mode='a', header=False, index=False)
#     else:
#         csvdata.to_csv(outputcsv, header=True, index=False)

#     ## TODO: output saving to csv for non-binary
# else:
#     test_auroc = roc_auc_score(true_y_test, pred_y_test)
#     test_auprc = average_precision_score(true_y_test, pred_y_test)
#     print("Test AUROC and AUPRC values are ", np.round(test_auroc, 4), np.round(test_auprc, 4))
#     fpr_roc, tpr_roc, thresholds_roc = roc_curve(true_y_test, pred_y_test, drop_intermediate=False)
#     precision_prc, recall_prc, thresholds_prc = precision_recall_curve(true_y_test, pred_y_test)
#     # interpolation in ROC
#     mean_fpr = np.linspace(0, 1, 100)
#     tpr_inter = np.interp(mean_fpr, fpr_roc, tpr_roc)
#     mean_fpr = np.round(mean_fpr, decimals=2)
#     print("Sensitivity at 90%  specificity is ", np.round(tpr_inter[np.where(mean_fpr == 0.10)], 2))

#     csvdata = {
#         'hp': json.dumps(vars(args)),
#         'Initial_seed': randomSeed,  # this is being done so its easier to differentiate each line in the final csv file
#         'outcome_rate': np.round(sum(outcome_df["outcome"].values) / len(outcome_df), decimals=4),
#         'AUROC': test_auroc,
#         'AUPRC': test_auprc,
#         'Sensitivity': tpr_inter[np.where(mean_fpr == 0.10)],
#         'git': args.git,
#         'name': args.nameinfo,
#         'target': args.task,
#         'evaltime': datetime.now().strftime("%y-%m-%d-%H:%M:%S")
#     }

#     csvdata = pd.DataFrame(csvdata)
#     outputcsv = os.path.join('/output/', args.outputcsv)
#     if (os.path.exists(outputcsv)):
#         csvdata.to_csv(outputcsv, mode='a', header=False, index=False)
#     else:
#         csvdata.to_csv(outputcsv, header=True, index=False)
