# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 10:42:57 2019

@author: Jiang
"""

import multiprocessing as mp
import numpy as np
import scipy.stats as ss


def mean_sqaure_rows(data_matrix):
    group_means = np.mean(data_matrix, axis=1)
    return np.var(group_means, ddof=1) * data_matrix.shape[1]


def mean_sqaure_within(data_matrix):
    var_within = []
    for j in range(data_matrix.shape[0]):
        target_values = data_matrix[j, :]
        var_within.append(np.var(target_values, ddof=1))
    return np.mean(var_within)


def mean_sqaure_columns(data_matrix):
    group_means = np.mean(data_matrix, axis=0)
    return np.var(group_means, ddof=1) * data_matrix.shape[0]


def fast_icc_analysis(tag, data_matrix, one_way=True, absolute=True, confidence_interval=0.95, single=True):
    icc = ICC(data_matrix)
    if one_way:
        score = icc.single_score_one_way_random()
        model = '.'
    else:
        if absolute:
            score = icc.single_score_two_way_random_absolute()
            model = 'A'
        else:
            score = icc.single_score_two_way_random_relative()
            model = 'C'
    lower_limit, higher_limit = icc.confidence_interval_limits(confidence_interval, score, model, single)
    return tag, score, lower_limit, higher_limit


class ICC_Parallel:
    def __init__(self, one_way=False, absolute=True, confidence_interval=0.95, single=True):
        self.one_way = one_way
        self.absolute = absolute
        self.confidence_interval = confidence_interval
        self.single = single
        self.tag_matrix_parameters_tuple_list = []

    def feed(self, tag, data_matrix):
        self.tag_matrix_parameters_tuple_list.append(
            (tag, data_matrix, self.one_way, self.absolute, self.confidence_interval, self.single))

    def excecute(self):
        pool = mp.Pool(mp.cpu_count() - 1)
        tag_icc_analysis_results_tuple_list = pool.starmap_async(
            fast_icc_analysis,
            self.tag_matrix_parameters_tuple_list).get()
        pool.close()
        pool.join()
        return tag_icc_analysis_results_tuple_list


class ICC:
    def __init__(self, data_matrix):
        self.n, self.k = data_matrix.shape
        self.msr = mean_sqaure_rows(data_matrix)
        self.msc = mean_sqaure_columns(data_matrix)
        self.msw = mean_sqaure_within(data_matrix)
        self.mse = (self.msw * self.n - self.msc) / (self.n - 1)

    def single_score_two_way_random_absolute(self):
        return (self.msr - self.mse) / (self.msr + (self.k - 1) * self.mse + self.k / self.n * (self.msc - self.mse))

    def single_score_two_way_random_relative(self):
        return (self.msr - self.mse) / (self.msr + (self.k - 1) * self.mse)

    def single_score_one_way_random(self):
        if (self.msr + (self.k - 1) * self.msw) == 0:
            return 1
        return (self.msr - self.msw) / (self.msr + (self.k - 1) * self.msw)

    def confidence_interval_limits(self, ci, icc_score, model='C', single=True):
        if self.msw ==0 or self.mse == 0:
            return (1,1)
        if model == 'A':
            a = self.k * icc_score / self.n / (1 - icc_score)
            b = 1 + self.k * icc_score * (self.n - 1) / self.n / (1 - icc_score)
            v = (a * self.msc + b * self.mse) ** 2 / (
                        (a * self.msc) ** 2 / (self.k - 1) + (b * self.mse) ** 2 / (self.n - 1) / (self.k - 1))
            dfn = self.n - 1
            dfd = v
            f_star = ss.f.ppf(1 - (1 - ci) / 2, dfn, dfd)
            if single:
                lower_limit = self.n * (self.msr - f_star * self.mse) / (f_star * (
                            self.k * self.msc + (self.k * self.n - self.k - self.n) * self.mse) + self.n * self.msr)
            else:
                lower_limit = self.n * (self.msr - f_star * self.mse) / (
                            f_star * (self.msc - self.mse) + self.n * self.msr)
            dfn = v
            dfd = self.n - 1
            f_star = ss.f.ppf(1 - (1 - ci) / 2, dfn, dfd)
            if single:
                upper_limit = self.n * (f_star * self.msr - self.mse) / (self.k * self.msc + (
                            self.k * self.n - self.k - self.n) * self.mse + self.n * f_star * self.msr)
            else:
                upper_limit = self.n * (f_star * self.msr - self.mse) / (
                            self.msc - self.mse + self.n * f_star * self.msr)
        else:

            if model == 'C':
                f_obs = self.msr / self.mse
                dfn = self.n - 1
                dfd = (self.n - 1) * (self.k - 1)
                f_star = ss.f.ppf(1 - (1 - ci) / 2, dfn, dfd)
                fl = f_obs / f_star
                dfn = (self.n - 1) * (self.k - 1)
                dfd = self.n - 1
                f_star = ss.f.ppf(1 - (1 - ci) / 2, dfn, dfd)
                fu = f_obs * f_star
            else:
                f_obs = self.msr / self.msw
                dfn = self.n - 1
                dfd = self.n * (self.k - 1)
                f_star = ss.f.ppf(1 - (1 - ci) / 2, dfn, dfd)
                fl = f_obs / f_star
                dfn = self.n * (self.k - 1)
                dfd = self.n - 1
                f_star = ss.f.ppf(1 - (1 - ci) / 2, dfn, dfd)
                fu = f_obs * f_star
            if single:
                lower_limit = (fl - 1) / (fl + (self.k - 1))
                upper_limit = (fu - 1) / (fu + (self.k - 1))
            else:
                lower_limit = 1 - 1 / fl
                upper_limit = 1 - 1 / fu

        return (lower_limit, upper_limit)
