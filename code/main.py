import pandas as pd

from modeling import modeling_pipeline
from preprocessing import feature_preparations, feature_filtering
import os

from repeatability_analysis import icc_test_retest, icc_perturbation




if __name__ == '__main__':
    raw_feature_directory = '../raw_features'
    preprocessed_feature_directory = '../preprocessed_features'
    feature_repeatability_directory = '../feature_repeatability'
    modeling_directory = '../modeling'
    inclusion_keywords = ['shape','32_binCount']

    # # feature preprocessing
    # clinical_feature_filepath = os.path.join(raw_feature_directory, 'clinical_features.csv')
    # radiomics_feature_filepaths = {
    #     'training': os.path.join(raw_feature_directory, 'training_features.csv'),
    #     'testing': os.path.join(raw_feature_directory, 'testing_features.csv'),
    #     'training_perturbation': os.path.join(raw_feature_directory, 'training_perturbation_features.csv'),
    #     'testing_perturbation': os.path.join(raw_feature_directory, 'testing_perturbation_features.csv'),
    #     'test_retest': os.path.join(raw_feature_directory, 'test_retest_features.csv'),
    # }
    # feature_preparations(radiomics_feature_filepaths, clinical_feature_filepath, preprocessed_feature_directory,
    #                      inclusion_keywords=inclusion_keywords)

    # feature repeatability analysis
    # for label in ['training', 'testing']:
    #     feature_table = pd.read_csv(os.path.join(raw_feature_directory, label+'_perturbation_features.csv'),
    #                                 header=[0, 1],
    #                                 index_col=0)
    #     included_features = feature_filtering(feature_table.index, inclusion_keywords)
    #     feature_table = feature_table.loc[included_features,:]
    #     export_filepath = os.path.join(feature_repeatability_directory, label+'_perturbation_icc.csv')
    #     icc_test_retest(feature_table, export_filepath)
    # feature_table = pd.read_csv(os.path.join(raw_feature_directory, 'test_retest_features.csv'),
    #                             header=[0, 1],
    #                             index_col=0)
    # included_features = feature_filtering(feature_table.index, inclusion_keywords)
    # feature_table = feature_table.loc[included_features, :]
    # export_filepath = os.path.join(feature_repeatability_directory, 'test_retest_icc.csv')
    # icc_test_retest(feature_table, export_filepath)

    selected_feature_number = 5
    endpoint = 'pCR'
    icc_thresholds = [0, 0.5, 0.75, 0.9, 0.95]
    bs_iterations = 1000
    random_state = 100
    icc_filenames = ['training_perturbation_icc.csv', 'test_retest_icc.csv']
    test_names = ['perturbation', 'test_retest']
    for icc_threshold in icc_thresholds:
        model_pipeline = modeling_pipeline(preprocessed_feature_directory,feature_repeatability_directory, modeling_directory,
                                                 number_selected=selected_feature_number,
                                                 icc_threshold=icc_threshold,endpoint=endpoint,
                                                 bs_iterations=bs_iterations, random_state=random_state)
        for icc_filename, test_name in zip(icc_filenames, test_names):
            model_pipeline.execute(icc_filename, raw_feature_directory, label=test_name+'_icc'+str(icc_threshold))
