import numpy as np
import pandas as pd
import os
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score


def permutation_p_value(measurement_function, left_data, right_data, permutation_iteration=100, random_seed=100):
    random_state = np.random.RandomState(random_seed)
    x = left_data[:, 0]
    y = left_data[:, 1]
    left_measurement = measurement_function(x,y)
    x = right_data[:, 0]
    y = right_data[:, 1]
    right_measurement = measurement_function(x,y)
    original_measurement_diff = np.abs(left_measurement - right_measurement)
    diff_distribution = []
    for i in range(permutation_iteration):
        left_choice_mask = random_state.randint(2, size=left_data.shape[0]).astype(bool)
        left_data_shuffles = np.concatenate([left_data[left_choice_mask,:], right_data[~left_choice_mask,:]], axis=0)
        right_data_shuffles = np.concatenate([left_data[~left_choice_mask,:], right_data[left_choice_mask,:]], axis=0)
        x = left_data_shuffles[:, 0]
        y = left_data_shuffles[:, 1]
        left_measurement = measurement_function(x, y)
        x = right_data_shuffles[:, 0]
        y = right_data_shuffles[:, 1]
        right_measurement = measurement_function(x, y)
        measurement_diff = np.abs(left_measurement - right_measurement)
        diff_distribution.append(measurement_diff)
    diff_distribution = np.array(diff_distribution)
    rejection_rate = np.sum(diff_distribution > original_measurement_diff) + np.sum(diff_distribution < -original_measurement_diff)
    rejection_rate = rejection_rate/len(diff_distribution)
    return rejection_rate, original_measurement_diff, diff_distribution

def classification_performance_summary(data_directory, test, performance_filename, bs_performance_filename):
    test1_performance = pd.read_csv(os.path.join(data_directory, test, performance_filename), index_col=0)
    test1_bs_performance = pd.read_csv(os.path.join(data_directory, test, bs_performance_filename), index_col=0)
    
    test1_summary = {}
    for group_name, single_group_bs_performance in test1_bs_performance.groupby('Cohort'):
        lower = single_group_bs_performance[test1_performance.index.values].quantile(q=0.025)
        higher = single_group_bs_performance[test1_performance.index.values].quantile(q=0.975)
        lower.name = 'Lower 95CI'
        higher.name = 'Higher 95CI'
        middle = test1_performance[group_name]
        middle.name = 'Original'
        single_group_summary = pd.concat([middle, lower, higher], axis=1)
        test1_summary[group_name] = single_group_summary
    test1_summary = pd.concat(test1_summary, axis=0)
    return test1_summary



def classification_performance_comparison(data_directory, test1, test2, prediction_filename, permutation_iteration=100,
                                random_seed=100):
    test1_predictions = pd.read_csv(os.path.join(data_directory, test1, prediction_filename), index_col=0)
    test2_predictions = pd.read_csv(os.path.join(data_directory, test2, prediction_filename), index_col=0)
    rejection_rates = {}
    diff_distributions = {}
    original_diffs = {}
    for metric_name in ['AUC', 'Accuracy', 'Precision', 'Recall']:
        if metric_name == 'AUC':
            scorer = roc_auc_score
            input_titles = ['Ground truth', 'Probability']
        elif metric_name == 'Accuracy':
            scorer = accuracy_score
            input_titles = ['Ground truth', 'Prediction']
        elif metric_name == 'Precision':
            scorer = precision_score
            input_titles = ['Ground truth', 'Prediction']
        elif metric_name == 'Recall':
            scorer = recall_score
            input_titles = ['Ground truth', 'Prediction']
        else:
            continue
        rejection_rate, original_diff, diff_distribution = permutation_p_value(scorer,
                                                                test1_predictions[input_titles].values,
                                                                test2_predictions[input_titles].values,
                                                                permutation_iteration=permutation_iteration,
                                                                random_seed=random_seed)
        rejection_rates[metric_name] = rejection_rate
        diff_distributions[metric_name] = pd.Series(diff_distribution)
        original_diffs[metric_name] = original_diff
    rejection_rates = pd.Series(rejection_rates)
    original_diffs = pd.Series(original_diffs)
    diff_distributions = pd.concat(diff_distributions, axis=1)
    return rejection_rates, original_diffs, diff_distributions

def classification_performance_analysis_pipeline(data_directory, perturbation_folders, test_retest_folders, export_directory,
                                                 permutation_iteration=100,random_seed=100):
    performance_filename = 'performance.csv'
    bs_performance_filename = 'bs_performance.csv'
    prediction_filenames = {
        'Training': 'training_predictions.csv',
        'Testing': 'testing_predictions.csv'
    }
    diff_distributions_all = {}
    original_diffs_all = {}

    perturbation_performance_summary = {}
    for i, perturbation_folder in enumerate(perturbation_folders):
        perturbation_performance_summary[perturbation_folder] = classification_performance_summary(data_directory, perturbation_folder, performance_filename,
                                           bs_performance_filename)
        if i > 0:
            for cohort, prediction_filename in prediction_filenames.items():
                rejection_rates, original_diffs, diff_distributions = classification_performance_comparison(data_directory,
                                                                                                            perturbation_folders[0],
                                                                                                            perturbation_folder,
                                                                                                            prediction_filename,
                                                                                                            permutation_iteration=permutation_iteration,
                                                                                                            random_seed=random_seed)
                perturbation_performance_summary[perturbation_folder].loc[
                    (cohort, rejection_rates.index.values), 'longitudinal_p_value'] = rejection_rates.values

                diff_distributions_all[perturbation_folders[0]+'_'+perturbation_folder+'_'+cohort] = diff_distributions
                original_diffs_all[perturbation_folders[0]+'_'+perturbation_folder+'_'+cohort] = original_diffs


    test_retest_performance_summary = {}
    for i, test_retest_folder in enumerate(test_retest_folders):
        test_retest_performance_summary[test_retest_folder] = classification_performance_summary(data_directory,
                                                                                                 test_retest_folder,
                                                                                                 performance_filename,
                                                                                                 bs_performance_filename)
        if i > 0:
            for cohort, prediction_filename in prediction_filenames.items():
                rejection_rates, original_diffs, diff_distributions = classification_performance_comparison(data_directory,
                                                                                                            test_retest_folders[0],
                                                                                                            test_retest_folder,
                                                                                                            prediction_filename,
                                                                                                            permutation_iteration=permutation_iteration,
                                                                                                            random_seed=random_seed)
                test_retest_performance_summary[test_retest_folder].loc[
                    (cohort, rejection_rates.index.values), 'longitudinal_p_value'] = rejection_rates.values

                diff_distributions_all[test_retest_folders[0]+'_'+test_retest_folder+'_'+cohort] = diff_distributions
                original_diffs_all[test_retest_folders[0]+'_'+test_retest_folder+'_'+cohort] = original_diffs

    for perturbation_folder, test_retest_folder in zip(perturbation_folders, test_retest_folders):
        for cohort, prediction_filename in prediction_filenames.items():
            rejection_rates, original_diffs, diff_distributions = classification_performance_comparison(data_directory,
                                                                                        perturbation_folder,
                                                                                        test_retest_folder,
                                                                                        prediction_filename,
                                                  permutation_iteration=permutation_iteration,
                                                  random_seed=random_seed)
            test_retest_performance_summary[test_retest_folder].loc[(cohort,rejection_rates.index.values),'parallel_p_value'] = rejection_rates.values
            diff_distributions_all[perturbation_folder+'_'+test_retest_folder+'_'+cohort] = diff_distributions
            original_diffs_all[perturbation_folder+'_'+test_retest_folder+'_'+cohort] = original_diffs
        combined_summary = pd.concat([perturbation_performance_summary[perturbation_folder],
                                      test_retest_performance_summary[test_retest_folder]],
                                     axis=1, keys=['Perturbation','Test_retest'])
        combined_summary.to_csv(os.path.join(export_directory,
                                               '{0}_{1}_classification_summary.csv'.format(perturbation_folder, test_retest_folder)))

    diff_distributions_all = pd.concat(diff_distributions_all, axis=1)
    diff_distributions_all.to_csv(os.path.join(export_directory,
                                               'classification_diff_distribution.csv'))
    original_diffs_all = pd.concat(original_diffs_all, axis=1)
    original_diffs_all.to_csv(os.path.join(export_directory,
                                           'classification_diff_original.csv'))

def prediction_variation_bias_analysis(data_directory, perturbation_folders, test_retest_folders,
                                                 export_directory):
    results = {}
    for model_name in perturbation_folders+test_retest_folders:
        for cohort_name, cohort in zip(['Training','Testing','Test-retest'],['training_perturbation', 'testing_perturbation', 'test_retest']):
            model_performance_directory = os.path.join(data_directory, model_name)
            training_predictions = pd.read_csv(os.path.join(model_performance_directory, cohort+'_predictions.csv'), index_col=0)
            training_predictions_mean = np.mean(training_predictions.values, axis=1)
            training_predictions_std = np.std(training_predictions.values, axis=1)
            stats = pearsonr(training_predictions_std,training_predictions_mean)
            results[model_name+'_'+cohort_name] = pd.Series(stats, index=['R','p'])
    results = pd.concat(results, axis=1).T
    results.to_csv(os.path.join(export_directory, 'model_variation_bias.csv'))

def icc_correlation(icc_comparison_filepath):
    icc_comparison = pd.read_csv(icc_comparison_filepath)
    perturbation_iccs = icc_comparison.iloc[:,0].values
    test_retest_iccs = icc_comparison.iloc[:,1].values
    correlation, p = pearsonr(test_retest_iccs, perturbation_iccs)
    print(correlation)
    print(p)
    # c_index = concordance_index(perturbation_iccs,test_retest_iccs)
    # print(c_index)



if __name__ == '__main__':
    data_directory = '..//model_performance'
    perturbation_folders = ['perturbation_icc0','perturbation_icc0.5','perturbation_icc0.75','perturbation_icc0.9','perturbation_icc0.95']
    test_retest_folders = ['test_retest_icc0','test_retest_icc0.5','test_retest_icc0.75','test_retest_icc0.9','test_retest_icc0.95']
    export_directory = '../statistical_analysis'
    permutation_iteration = 1000
    random_seed = 100
    classification_performance_analysis_pipeline(data_directory, perturbation_folders, test_retest_folders,
                                                 export_directory,
                                                 permutation_iteration=permutation_iteration,
                                                 random_seed=random_seed)

    prediction_variation_bias_analysis(data_directory, perturbation_folders, test_retest_folders,
                                       export_directory)

    icc_comparison_filepath = os.path.join(export_directory, 'icc_comparison.csv')
    icc_correlation(icc_comparison_filepath)




        
    
        
        
    
    




