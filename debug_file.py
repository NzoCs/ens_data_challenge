# Import necessary libraries
import pandas as pd

from sksurv.util import Surv

from ens_data_challenge.globals import TRAIN_CLINICAL_DATA_PATH, TRAIN_MOLECULAR_DATA_PATH, TRAIN_TARGET_PATH, TEST_CLINICAL_DATA_PATH, TEST_MOLECULAR_DATA_PATH
clinical_data_train = pd.read_csv(TRAIN_CLINICAL_DATA_PATH)
clinical_data_eval = pd.read_csv(TEST_CLINICAL_DATA_PATH)

# Molecular Data
molecular_data_train = pd.read_csv(TRAIN_MOLECULAR_DATA_PATH)
molecular_data_eval = pd.read_csv(TEST_MOLECULAR_DATA_PATH)

target_df = pd.read_csv(TRAIN_TARGET_PATH)

# Preview the data
clinical_data_train.head()

from ens_data_challenge.preprocess.preprocessor import Preprocessor

preprocessor = Preprocessor()

clinical_data_train, cyto_df_train = preprocessor.get_cyto_features_and_df(clinical_data_train)
clinical_data_eval, cyto_df_eval = preprocessor.get_cyto_features_and_df(clinical_data_eval)

(
    clinical_preprocess_train,
    clinical_data_eval, 
    molecular_preprocess_train, 
    molecular_preprocess_eval, 
    cyto_struct_preprocess_train, 
    cyto_struct_preprocess_eval,
    targets_preprocess
  ) = preprocessor.fit_transform(
    clinical_data_train=clinical_data_train,
    molecular_data_train=molecular_data_train,
    clinical_data_test=clinical_data_eval,
    molecular_data_test=molecular_data_eval,
    cyto_struct_train=cyto_df_train,
    cyto_struct_test=cyto_df_eval,
    targets=target_df,
)

from ens_data_challenge.feature_engineering.feat_eng_helper import FeatureEngineerHelper
from ens_data_challenge.types import CytoColumns, MolecularColumns, CytoStructColumns

fe = FeatureEngineerHelper()

clinical_data_train = fe.add_mol_encoding(clinical_data_train, molecular_preprocess_train, col='GENE', method='confidence_weighted', apply_effect_weighting=True)
clinical_data_eval = fe.add_mol_encoding(clinical_data_eval, molecular_preprocess_eval, col='GENE', method='confidence_weighted', apply_effect_weighting=True)