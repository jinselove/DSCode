import pandas as pd
import numpy as np


class InfoGain:

    def __init__(self, info_type="entropy", **params):
        """

        :info_type: either "entropy" or "gini"
        :split_mode: the way of splitting data, supports "random", "medium"
        :split_threshold: a dictionary contains split threshold for each column in train_X
        """
        self._info_type = info_type
        self._params = params

    @property
    def info_type(self):
        return self._info_type

    @property
    def params(self):
        return self._params

    @staticmethod
    def get_random_split_threshold(train_x):
        """Generate random split threshold for all cols

        :param train_x: training data, [dataframe]
        :return: a dictionary of random split threshold for all cols
        """
        split_threshold_dict = {}
        for col in train_x:
            split_threshold_dict[col] = train_x[col].sample(n=1).iloc[0]

        return split_threshold_dict

    def get_split_threshold(self, train_x):
        """

        :train_x: training data X, [dataframe]
        :return: a dictionary contains split threshold for each variable
        """
        try:
            split_mode = self._params["split_mode"]
            if split_mode == "random":
                split_threshold_dict = self.get_random_split_threshold(train_x)
            else:
                print("Error: split mode {0} not supported yet".format(split_mode))
                exit(-1)
        except KeyError:
            try:
                split_threshold_dict = self._params["split_threshold"]
            except KeyError:
                print("Error: if split_mode is not given, then split_threshold has to be specified")
                exit(-1)

        return split_threshold_dict

    @staticmethod
    def entropy(prob_list):
        """

        :param prob_list: a list contains the probability of all classes
        :return: the entropy
        """
        result = 0
        for prob in prob_list:
            result -= prob * np.log2(prob)
        return result

    def cal_target_entropy(self, df_target):
        """

        :param df_target: the target column
        :return: the entropy of the target column
        """
        class_prob = df_target.value_counts() / df_target.shape[0]
        target_entropy = self.entropy(list(class_prob.values))

        return target_entropy

    def cal_col_entropy(self, df_col, df_target, split_threshold):
        """calculate the entropy of column

        :param df_col: column that the entropy is calculated for, [df_series]
        :param df_target: target column, [df_series]
        :param split_threshold: split threshold for this column, [numeric]
        :return: entropy of the column
        """
        positive_index = df_col[df_col >= split_threshold].index
        negative_index = df_col.index.difference(positive_index)

        target_entropy_positive = self.cal_target_entropy(df_target.iloc[positive_index])
        target_entropy_negative = self.cal_target_entropy(df_target.iloc[negative_index])

        p_positive = len(positive_index) / len(df_col.index)
        variable_entropy = p_positive * target_entropy_positive + (1 - p_positive) * target_entropy_negative

        return variable_entropy

    def cal_entropy(self, train_x, train_y, split_threshold_dict):
        """calculate information gain for each column using entropy

        :param train_x: training data X, [dataframe]
        :param train_y: target col, [df_series]
        :param split_threshold_dict: split threshold dict for all features
        :return: a dictionary of information gain
        """
        target_entropy = self.cal_target_entropy(train_y)
        info_gain_dict = {}
        for col in train_x:
            info_gain_dict[col] = target_entropy - self.cal_col_entropy(train_x[col],
                                                                        train_y,
                                                                        split_threshold_dict[col])

        return info_gain_dict

    @staticmethod
    def gini(prob_list):
        """

        :param prob_list: a list contains the probability of all classes
        :return: the entropy
        """
        result = 0
        for prob in prob_list:
            result += prob**2
        return 1-result

    def cal_target_gini(self, df_target):
        """

        :param df_target: the target column
        :return: the entropy of the target column
        """
        class_prob = df_target.value_counts() / df_target.shape[0]
        target_gini = self.gini(list(class_prob.values))

        return target_gini

    def cal_col_gini(self, df_col, df_target, split_threshold):
        """calculate the Gini index of a column

        :param df_col: column that the Gini index is calculated for, [df_series]
        :param df_target: target column, [df_series]
        :param split_threshold: split threshold for this column, [numeric]
        :return: gini index of the column
        """
        positive_index = df_col[df_col >= split_threshold].index
        negative_index = df_col.index.difference(positive_index)

        target_gini_positive = self.cal_target_gini(df_target.iloc[positive_index])
        target_gini_negative = self.cal_target_gini(df_target.iloc[negative_index])

        p_positive = len(positive_index) / len(df_col.index)
        variable_gini = p_positive * target_gini_positive + (1 - p_positive) * target_gini_negative

        return variable_gini

    def cal_gini(self, train_x, train_y, split_threshold_dict):
        """calculate information gain for each column using Gini index

        :param train_x: training data X, [dataframe]
        :param train_y: target col, [df_series]
        :param split_threshold_dict: split threshold dict for all features
        :return: a dictionary of information gain
        """
        info_gain_dict  = {}
        for col in train_x:
            info_gain_dict[col] = self.cal_col_gini(train_x[col],
                                                    train_y,
                                                    split_threshold_dict[col])

        return info_gain_dict

    def cal_info_gain(self, train_x, train_y):
        """calculate information gain for each column

        :return: a dictionary of information gain
        """
        split_threshold_dict = self.get_split_threshold(train_x)
        if self._info_type == "entropy":
            info_gain_dict = self.cal_entropy(train_x, train_y, split_threshold_dict)
        elif self._info_type == "gini":
            info_gain_dict = self.cal_gini(train_x, train_y, split_threshold_dict)
        else:
            print("Error: info_type [{0}] not supported".format(self._info_type))
            exit(-1)

        return info_gain_dict
