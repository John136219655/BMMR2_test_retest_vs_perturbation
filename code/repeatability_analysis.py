import icc_anova as ia
import pandas as pd


def icc_calculation_pipeline(feature_table, num_raters,
                             one_way=False, absolute=True, confidence_interval=0.95, single=True):
    icc_parallel = ia.ICC_Parallel(one_way=one_way, absolute=absolute,
                                   confidence_interval=confidence_interval, single=single)
    feature_labels = feature_table.index.values

    for feature_label in feature_labels:
        single_feature_data = feature_table.loc[feature_label,:].values.reshape((-1, num_raters))
        # single_feature_data = perturbation_feature_table[feature_label].values.reshape((-1, 40))
        index = 0
        # for train_index, test_index in cv.split(single_feature_data):
        #     resample_feature_data = single_feature_data[train_index, :]
        icc_parallel.feed((feature_label, index), single_feature_data)
        index += 1
    icc_results = icc_parallel.excecute()
    icc_scores = dict()
    icc_lower_limits = dict()
    icc_higher_limits = dict()
    for tag, icc_score, lower_limit, higher_limit in icc_results:
        feature_label = tag[0]
        icc_scores[feature_label]=icc_score
        icc_lower_limits[feature_label]=lower_limit
        icc_higher_limits[feature_label]=higher_limit
    icc_results = pd.concat([pd.Series(icc_scores, name = 'ICC'), pd.Series(icc_lower_limits, name='Lower'),
                             pd.Series(icc_higher_limits, name='Higher')], axis=1)
    return icc_results

def icc_test_retest(feature_table, export_filepath):
    icc_results = icc_calculation_pipeline(feature_table, 2,
                             one_way=False, absolute=True, confidence_interval=0.95, single=True)
    icc_results.to_csv(export_filepath)

def icc_perturbation(feature_table, export_filepath):
    icc_results = icc_calculation_pipeline(feature_table, 2,
                             one_way=True, absolute=True, confidence_interval=0.95, single=True)
    icc_results.to_csv(export_filepath)
