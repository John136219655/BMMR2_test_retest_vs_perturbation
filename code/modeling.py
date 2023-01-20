import os
import pandas as pd
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import r_regression

import mrmr
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, roc_curve
from icc_anova import ICC, fast_icc_analysis

def model_performance(model, predictors, ground_truth):
    scores = {}
    predictions_proba = model.predict_proba(predictors)[:,1]
    predictions_binary = model.predict(predictors)
    for metric_name in ['AUC', 'Accuracy', 'Precision', 'Recall']:
        if metric_name == 'AUC':
            score = roc_auc_score(ground_truth, predictions_proba)
        elif metric_name == 'Accuracy':
            score = accuracy_score(ground_truth, predictions_binary)
        elif metric_name == 'Precision':
            score = precision_score(ground_truth, predictions_binary)
        elif metric_name == 'Recall':
            score = recall_score(ground_truth, predictions_binary)
        else:
            continue
        scores[metric_name] = score
    scores = pd.Series(scores)
    fpr, tpr, thresholds = roc_curve(ground_truth, predictions_proba)
    roc = pd.DataFrame([fpr, tpr, thresholds],index=['FPR','TPR','Threshold']).T
    return scores, roc

def bootstrap_model_performance(model, predictors, ground_truth, bs_iterations=100, random_seed=100):
    predictions_proba = model.predict_proba(predictors)[:, 1]
    predictions_binary = model.predict(predictors)
    random_state = np.random.RandomState(random_seed)
    bs_scores = []
    bs_rocs = []
    resampled_fpr = np.arange(0,1.05,0.05)
    for i in range(bs_iterations):
        # if i == 0:
        #     bs_index = np.arange(len(predictions_binary))
        # else:
        bs_index = random_state.choice(np.arange(len(predictions_binary)), len(predictions_binary), replace=True)
        bs_ground_truth = ground_truth[bs_index]
        bs_predictions_proba = predictions_proba[bs_index]
        bs_predictions_binary = predictions_binary[bs_index]
        scores = {}
        for metric_name in ['AUC', 'Accuracy', 'Precision', 'Recall']:
            if metric_name == 'AUC':
                score = roc_auc_score(bs_ground_truth, bs_predictions_proba)
            elif metric_name == 'Accuracy':
                score = accuracy_score(bs_ground_truth, bs_predictions_binary)
            elif metric_name == 'Precision':
                score = precision_score(bs_ground_truth, bs_predictions_binary)
            elif metric_name == 'Recall':
                score = recall_score(bs_ground_truth, bs_predictions_binary)
            else:
                continue
            scores[metric_name] = score
        scores = pd.Series(scores)
        scores = pd.concat([scores], axis=1).T
        scores['BS iteration'] = [i]*scores.shape[0]
        fpr, tpr, thresholds = roc_curve(bs_ground_truth, bs_predictions_proba, drop_intermediate=True)
        resampled_tpr = np.interp(resampled_fpr, fpr, tpr)
        roc = pd.DataFrame([resampled_tpr], index=['TPR']).T
        roc['FPR'] = resampled_fpr
        roc['BS iteration'] = [i]*roc.shape[0]
        bs_scores.append(scores)
        bs_rocs.append(roc)
    bs_rocs = pd.concat(bs_rocs, axis=0)
    bs_scores= pd.concat(bs_scores, axis=0)
    return bs_scores, bs_rocs

def model_perturbation_repeatability(feature_names, scaler, model, ground_truth, perturbation_feature_filepath):
    perturbation_feature = pd.read_csv(perturbation_feature_filepath, index_col=0, header=[0,1])
    perturbation_feature = perturbation_feature.loc[feature_names, (ground_truth.index.values, slice(None))]
    feature_values = perturbation_feature.values.T
    predictors = scaler.transform(feature_values)
    perturbation_predictions = model.predict_proba(predictors)[:, 1]
    perturbation_predictions = perturbation_predictions.reshape((-1, 40))
    _, icc_score, lower_limit, higher_limit = fast_icc_analysis(None, perturbation_predictions, one_way=True, absolute=True, confidence_interval=0.95,
                      single=True)
    prediction_icc = (icc_score, lower_limit, higher_limit)
    # prediction_icc = ICC(all_patient_perturbation_predictions).single_score_one_way_random()
    all_patient_perturbation_predictions = pd.DataFrame(perturbation_predictions, index=ground_truth.index)
    # perturbation_performance = {}
    # for perturbation_index in all_patient_perturbation_predictions.columns:
    #     scores, roc = model_performance(all_patient_perturbation_predictions[perturbation_index].values, ground_truth[valid_patient_ids].values)
    #     perturbation_performance[perturbation_index] = scores
    # perturbation_performance = pd.concat(perturbation_performance, axis=1).T
    return prediction_icc, all_patient_perturbation_predictions#, perturbation_performance

def model_test_retest_repeatability(feature_names, scaler, model, test_retest_feature_filepath):
    test_retest_feature = pd.read_csv(test_retest_feature_filepath, index_col=0, header=[0, 1])
    test_retest_feature = test_retest_feature.loc[feature_names, :]
    feature_values = test_retest_feature.values.T
    predictions = model.predict_proba(scaler.transform(feature_values))[:, 1]
    predictions = predictions.reshape((-1,2))
    _, icc_score, lower_limit, higher_limit = fast_icc_analysis(None, predictions,
                                                                one_way=True, absolute=True, confidence_interval=0.95,
                                                                single=True)
    prediction_icc = (icc_score, lower_limit, higher_limit)
    test_retest_predictions = pd.DataFrame(predictions, index=test_retest_feature.columns.get_level_values(0).unique())
    return prediction_icc, test_retest_predictions


class modeling_pipeline():
    def __init__(self, preprocessed_feature_directory, feature_repeatability_directory, results_directory, number_selected=10, icc_threshold=0.9,
                 icc_unit='absolute',
                 endpoint='pCR', bs_iterations=100, random_state=100):
        self.preprocessed_feature_directory = preprocessed_feature_directory
        self.feature_repeatability_directory = feature_repeatability_directory
        self.results_directory = results_directory
        self.number_selected = number_selected
        self.icc_threshold = icc_threshold
        self.icc_unit = icc_unit
        self.endpoint = endpoint
        self.bs_iterations = bs_iterations
        self.random_state = random_state
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)

        self.training_radiomics = None
        self.testing_radiomics = None
        self.training_clinical = None
        self.testing_clinical = None
        self.radiomics_icc = None
        self.selected_features = None
        self.scaler = None
        self.model = None

    def _read_features(self):
        self.training_radiomics = pd.read_csv(os.path.join(self.preprocessed_feature_directory, 'training_radiomics.csv'),
                                              index_col=0)
        self.testing_radiomics = pd.read_csv(os.path.join(self.preprocessed_feature_directory, 'testing_radiomics.csv'),
                                             index_col=0)
        self.training_clinical = pd.read_csv(os.path.join(self.preprocessed_feature_directory, 'training_clinical.csv'),
                                             index_col=0)
        self.testing_clinical = pd.read_csv(os.path.join(self.preprocessed_feature_directory, 'testing_clinical.csv'),
                                            index_col=0)

    def _read_radiomics_icc(self, icc_filename):
        self.radiomics_icc = pd.read_csv(os.path.join(self.feature_repeatability_directory, icc_filename),index_col=0)

    # def df_check_nan(self, df, time_point, train_test):
    #     A = df.isnull()
    #     original_patients = A.index.values.tolist()
    #
    #     if not A.values.any():
    #         print('{} {} good'.format(time_point, train_test))
    #         return df
    #     else:
    #         new_df = df.dropna(axis=1)
    #         new_patients = new_df.index.values.tolist()
    #         print('{} {} patients {}'.format(time_point, train_test, list(set(original_patients) - set(new_patients))))
    #         return new_df

    def _feature_selection_pipeline(self):
        ### feature robustness filtering
        if self.icc_unit == 'absolute':
            abs_icc_threshold = self.icc_threshold
        else:
            abs_icc_threshold = np.percentile(self.radiomics_icc['ICC'].values, self.icc_threshold * 100)
        robust_feature_name = self.radiomics_icc.index.values[self.radiomics_icc['ICC'] >= abs_icc_threshold]
        print('Absolute ICC threshold: {0}. {1} features remained'.format(abs_icc_threshold, len(robust_feature_name)))
        training_radiomics = self.training_radiomics[robust_feature_name]
        # volume dependency removal
        volume_correlation = r_regression(training_radiomics, self.training_radiomics['original_shape_MeshVolume'])
        selected_feature_names = robust_feature_name[np.abs(volume_correlation) < 0.6]
        training_radiomics = training_radiomics[selected_feature_names]

        training_labels = self.training_clinical[self.endpoint]
        selected_features = mrmr.mrmr_classif(X=training_radiomics, y=training_labels, K=self.number_selected)
        self.selected_features = selected_features

    def _model_training(self):
        training_radiomics = self.training_radiomics.loc[:, self.selected_features].values
        training_labels = self.training_clinical[self.endpoint].values
        self.scaler = StandardScaler()
        training_radiomics = self.scaler.fit_transform(training_radiomics)
        model = EasyEnsembleClassifier(base_estimator=LogisticRegression(), n_estimators=500)
        model.fit(training_radiomics, training_labels)
        self.model = model

    def _model_export(self):
        if self.model is None:
            print('The model has not been trained.')
            return
        ensemble_model_parameters = []
        for sub_model in self.model.estimators_:
            sub_model = sub_model['classifier']
            # print(sub_model)
            model_parameters = list(sub_model.coef_.flatten()) + list(sub_model.intercept_)
            model_parameters = pd.Series(model_parameters, index=list(self.selected_features) + ['intercept'])
            ensemble_model_parameters.append(model_parameters)
        ensemble_model_parameters.append(
            pd.Series(self.scaler.scale_, name='Scale', index=list(self.selected_features)))
        ensemble_model_parameters.append(pd.Series(self.scaler.mean_, name='Mean', index=list(self.selected_features)))
        ensemble_model_parameters.append(pd.Series(self.scaler.var_, name='Var', index=list(self.selected_features)))
        ensemble_model_parameters = pd.concat(ensemble_model_parameters, axis=1)
        return ensemble_model_parameters

    def _model_calibration(self):
        result = []
        training_radiomics = self.training_radiomics.loc[:, self.selected_features].values
        training_labels = self.training_clinical[self.endpoint].values
        training_radiomics = self.scaler.transform(training_radiomics)
        y_prob = self.model.predict_proba(training_radiomics)
        y_prob = y_prob[:, 1]
        prob_true, prob_pred = calibration_curve(training_labels, y_prob)
        calibration = pd.DataFrame([prob_true, prob_pred], index=['Prob true', 'Prob pred']).T
        calibration['Cohort'] = ['Training'] * calibration.shape[0]
        result.append(calibration)

        testing_radiomics = self.testing_radiomics.loc[:, self.selected_features].values
        testing_labels = self.testing_clinical[self.endpoint].values
        testing_radiomics = self.scaler.transform(testing_radiomics)
        y_prob = self.model.predict_proba(testing_radiomics)
        y_prob = y_prob[:, 1]
        prob_true, prob_pred = calibration_curve(testing_labels, y_prob)
        calibration = pd.DataFrame([prob_true, prob_pred], index=['Prob true', 'Prob pred']).T
        calibration['Cohort'] = ['Testing'] * calibration.shape[0]
        result.append(calibration)

        result = pd.concat(result, axis=0)
        return result

    def _model_generalizability(self, export_directory):
        training_radiomics = self.training_radiomics.loc[:, self.selected_features].values
        training_labels = self.training_clinical[self.endpoint].values
        training_radiomics = self.scaler.fit_transform(training_radiomics)
        training_performance, training_roc = model_performance(self.model, training_radiomics, training_labels)
        testing_radiomics = self.testing_radiomics.loc[:, self.selected_features].values
        testing_labels = self.testing_clinical[self.endpoint].values
        testing_radiomics = self.scaler.fit_transform(testing_radiomics)
        testing_performance, testing_roc = model_performance(self.model, testing_radiomics, testing_labels)
        performance = pd.concat([training_performance, testing_performance], axis=1, keys=['Training', 'Testing'])
        rocs = pd.concat([training_roc, testing_roc], axis=1, keys=['Training', 'Testing'])

        training_bs_performance, training_bs_roc = bootstrap_model_performance(self.model, training_radiomics,
                                                                               training_labels,
                                                                               bs_iterations=self.bs_iterations,
                                                                               random_seed=self.random_state)
        training_bs_performance['Cohort'] = ['Training'] * training_bs_performance.shape[0]
        training_bs_roc['Cohort'] = ['Training'] * training_bs_roc.shape[0]
        testing_bs_performance, testing_bs_roc = bootstrap_model_performance(self.model, testing_radiomics,
                                                                             testing_labels,
                                                                             bs_iterations=self.bs_iterations,
                                                                             random_seed=self.random_state)
        testing_bs_performance['Cohort'] = ['Testing'] * testing_bs_performance.shape[0]
        testing_bs_roc['Cohort'] = ['Testing'] * testing_bs_roc.shape[0]
        bs_performance = pd.concat([training_bs_performance, testing_bs_performance], axis=0)
        bs_roc = pd.concat([training_bs_roc, testing_bs_roc], axis=0)

        bs_performance.to_csv(os.path.join(export_directory, 'bs_performance.csv'))
        bs_roc.to_csv(os.path.join(export_directory, 'bs_roc.csv'))

        training_predictions = pd.DataFrame(
            [self.model.predict_proba(training_radiomics)[:, 1], self.model.predict(training_radiomics),
             training_labels],
            index=['Probability', 'Prediction', 'Ground truth'], columns=self.training_clinical.index).T
        training_predictions = pd.concat(
            [pd.DataFrame(training_radiomics, index=self.training_radiomics.index, columns=self.selected_features),
             training_predictions], axis=1)
        training_predictions.to_csv(os.path.join(export_directory, 'training_predictions.csv'))

        testing_predictions = pd.DataFrame(
            [self.model.predict_proba(testing_radiomics)[:, 1], self.model.predict(testing_radiomics),
             testing_labels],
            index=['Probability', 'Prediction', 'Ground truth'], columns=self.testing_clinical.index).T
        testing_predictions = pd.concat(
            [pd.DataFrame(testing_radiomics, index=self.testing_radiomics.index, columns=self.selected_features),
             testing_predictions],
            axis=1)
        testing_predictions.to_csv(os.path.join(export_directory, 'testing_predictions.csv'))

        return performance, rocs

    def _model_reliability(self, raw_feature_directory, export_directory):
        training_labels = self.training_clinical[self.endpoint]
        training_prediction_icc, training_perturbation_predictions = model_perturbation_repeatability(
            self.selected_features, self.scaler, self.model,
            training_labels, os.path.join(raw_feature_directory, 'training_perturbation_features.csv'))
        training_perturbation_predictions.to_csv(
            os.path.join(export_directory, 'training_perturbation_predictions.csv'))
        testing_labels = self.testing_clinical[self.endpoint]
        testing_prediction_icc, testing_perturbation_predictions = model_perturbation_repeatability(
            self.selected_features, self.scaler, self.model,
            testing_labels, os.path.join(raw_feature_directory, 'testing_perturbation_features.csv'))
        testing_perturbation_predictions.to_csv(
            os.path.join(export_directory, 'testing_perturbation_predictions.csv'))
        test_retest_prediction_icc, test_retest_predictions = model_test_retest_repeatability(self.selected_features,
                                                                                              self.scaler, self.model,
                                                                                              os.path.join(raw_feature_directory, 'test_retest_features.csv'))
        prediction_icc = pd.DataFrame([training_prediction_icc, testing_prediction_icc, test_retest_prediction_icc],
                                      index=['Training', 'Testing', 'Test-retest'],
                                      columns=['Score', 'Lower', 'Higher'])
        test_retest_predictions.to_csv(os.path.join(export_directory, 'test_retest_predictions.csv'))
        # perturbation_performance = pd.concat([training_prediction_icc, testing_prediction_icc], keys=['Training','Testing'], axis=1)
        return prediction_icc  # , perturbation_performance

    def _model_evaluation(self, raw_feature_directory, export_directory):
        performance, rocs = self._model_generalizability(export_directory)
        performance.to_csv(os.path.join(export_directory, 'performance.csv'))
        rocs.to_csv(os.path.join(export_directory, 'rocs.csv'))
        prediction_icc = self._model_reliability(raw_feature_directory, export_directory)
        prediction_icc.to_csv(os.path.join(export_directory, 'prediction_icc.csv'))

    def execute(self, icc_filename, raw_feature_directory, label=None):
        if label is not None:
            export_directory = os.path.join(self.results_directory, label)
        else:
            export_directory = self.results_directory
        if not os.path.exists(export_directory):
            os.mkdir(export_directory)
        self._read_features()
        self._read_radiomics_icc(icc_filename)
        self._feature_selection_pipeline()
        self.radiomics_icc.loc[self.selected_features, :].to_csv(
            os.path.join(export_directory, 'selected_feature_icc.csv'))
        self._model_training()
        model_parameters = self._model_export()
        model_parameters.to_csv(os.path.join(export_directory, 'model_parameters.csv'))
        calibration = self._model_calibration()
        calibration.to_csv(os.path.join(export_directory, 'model_calibration.csv'))
        self._model_evaluation(raw_feature_directory, export_directory)