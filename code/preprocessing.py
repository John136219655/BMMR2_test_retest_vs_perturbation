import numpy as np
import pandas as pd
import os

def feature_filtering(feature_names, inclusion_keywords):
    included_features = []
    for feature_name in feature_names:
        for keyword in inclusion_keywords:
            if keyword in feature_name:
                included_features.append(feature_name)
                continue
    return included_features


def feature_preparations(radiomics_feature_filepaths, clinical_feature_filepath,
                         export_directory, inclusion_keywords=None):
    clinical_feature_table = pd.read_csv(clinical_feature_filepath, index_col=0)
    clinical_feature_table.index = ['ACRIN-6698-{0}'.format(patient_id) for patient_id in clinical_feature_table.index]
    for feature_category, radiomics_feature_filepath in radiomics_feature_filepaths.items():
        radiomics_feature_table = pd.read_csv(radiomics_feature_filepath, index_col=0).T
        if inclusion_keywords is not None:
            included_features = feature_filtering(radiomics_feature_table.columns, inclusion_keywords)
            radiomics_feature_table = radiomics_feature_table[included_features]
        common_index = radiomics_feature_table.index.intersection(clinical_feature_table.index)
        radiomics_feature_table.loc[common_index,:].to_csv(os.path.join(export_directory, feature_category+'_radiomics.csv'))
        clinical_feature_table.loc[common_index, :].to_csv(os.path.join(export_directory, feature_category+'_clinical.csv'))




# if __name__ == '__main__':
    # clinical_feature_filepath = '../raw_features/clinical_features.csv'
    # radiomics_feature_filepaths = {
    #     'train': '../raw_features/training_features.csv',
    #     'test': '../raw_features/testing_features.csv'
    # }
    # export_directory = '../preprocessed_features'
    # feature_preparations(radiomics_feature_filepaths, clinical_feature_filepath, export_directory)

    # feature_table_filepath = r"Z:\Radiomics_Projects\BMMR2\raw_features\test_retest_features.csv"
    # feature_table = pd.read_csv(feature_table_filepath, index_col=0, header=[0,1])
    # feature_table = feature_table.swaplevel(axis=1)
    # feature_table = feature_table.loc[:,feature_table.columns.sortlevel(0)[0]]
    # feature_table.to_csv(r"Z:\Radiomics_Projects\BMMR2\raw_features\test_retest_features_swapped.csv")





