import fire
import pandas as pd
import sklearn.metrics as metrics
import os
import os


class preprocess_data:
    _output_path = ""
    _correlation_cutoff = 0.70

    def _create_output_path(self):
        if not(os.path.exists(self._output_path)):
            os.mkdir(self._output_path)

    def __init__(self, output_path):
        self._output_path = output_path
        self._create_output_path()

    def preprocess_descriptive_statistics(self, df_data, x_cols):
        df_descriptive_statistics = df_data[x_cols].describe().transpose()
        df_descriptive_statistics.to_csv(f"{self._output_path}/descriptive_statistics.csv", index=False)
        return df_descriptive_statistics

    def preprocess_impute_missing(self, df_data, x_cols):
        df_data_imputed = df_data.copy()
        df_impute_parameters = pd.DataFrame()
        for col in x_cols:
            col_mean = df_data[col].mean()
            df_data_imputed[col] = df_data[col].fillna(col_mean)
            df_impute_parameters_col = pd.DataFrame({"variable": [col], "impute_value": [col_mean]})
            df_impute_parameters = pd.concat([df_impute_parameters, df_impute_parameters_col])
        df_impute_parameters.to_csv(f"{self._output_path}/impute_missing_parameters.csv", index=False)
        return df_data_imputed

    def preprocess_compute_bivariate_analysis(self, df_data, x_cols, y_col):
        pd_bivariate_analysis = pd.DataFrame()
        for col in x_cols:
            auc_col = metrics.roc_auc_score(df_data[y_col], df_data[col])
            if (auc_col < 0.5): auc_col = (1 - auc_col)
            pd_bivariate_analysis_col = pd.DataFrame({"variable": [col], "bivariate_auc": [auc_col]})
            pd_bivariate_analysis = pd.concat([pd_bivariate_analysis, pd_bivariate_analysis_col])
        pd_bivariate_analysis.to_csv(f"{self._output_path}/bivariate_analysis.csv", index=False)
        return pd_bivariate_analysis

    def preprocess_compute_correlation_pairs(self, df_data, x_cols):
        corr_matrix = df_data[x_cols].corr()
        corr_matrix_abs = corr_matrix.abs()
        so = corr_matrix_abs.unstack()
        df_corr_pairs_abs = pd.DataFrame(so).reset_index()
        df_corr_pairs_abs.columns = ["variable_1", "variable_2", "corr"]
        df_corr_pairs_abs = df_corr_pairs_abs[df_corr_pairs_abs["corr"] < 1]
        df_corr_pairs_abs_cutoff = df_corr_pairs_abs[df_corr_pairs_abs["corr"] >= self._correlation_cutoff]
        return df_corr_pairs_abs_cutoff

    def _find_variable_bivariate_auc(self, df_bivariate_analysis, variable_name):
        return df_bivariate_analysis[df_bivariate_analysis["variable"] == variable_name]["bivariate_auc"].iloc[0]

    def _find_bivariate_auc_high_correlation_pairs(self, df_bivariate_analysis, df_corr_pairs_abs_cutoff):
        df_corr_pairs_abs_cutoff_bivariate = df_corr_pairs_abs_cutoff.copy()
        auc_1_list, auc_2_list = [], []
        for index, row in df_corr_pairs_abs_cutoff.iterrows():
            auc_1_list.append(self._find_variable_bivariate_auc(df_bivariate_analysis, row["variable_1"]))
            auc_2_list.append(self._find_variable_bivariate_auc(df_bivariate_analysis, row["variable_2"]))
        df_corr_pairs_abs_cutoff_bivariate["bivariate_auc_1"] = auc_1_list
        df_corr_pairs_abs_cutoff_bivariate["bivariate_auc_2"] = auc_2_list
        return df_corr_pairs_abs_cutoff_bivariate

    def _filter_high_correlation_pairs(self, df_corr_pairs_abs_cutoff_bivariate):
        vars_selected = []
        for index, row in df_corr_pairs_abs_cutoff_bivariate.iterrows():
            var_selected = row["variable_1"]
            if (row["bivariate_auc_2"] > row["bivariate_auc_1"]):
                var_selected = row["variable_2"]
            vars_selected.append(var_selected)
        return list(set(vars_selected))

    def preprocess_clean_correlations(self, df_data, x_cols, y_col, df_corr_pairs_abs_cutoff, df_bivariate_analysis):
        df_corr_pairs_abs_cutoff_bivariate = self._find_bivariate_auc_high_correlation_pairs(df_bivariate_analysis, df_corr_pairs_abs_cutoff)
        x_cols_high_correlation = list(set(df_corr_pairs_abs_cutoff_bivariate["variable_1"] + df_corr_pairs_abs_cutoff_bivariate["variable_2"]))
        x_cols_low_correlation = [x for x in x_cols if x not in x_cols_high_correlation]

        x_cols_final = list(set(self._filter_high_correlation_pairs(df_corr_pairs_abs_cutoff_bivariate) + x_cols_low_correlation))
        df_vars_final = pd.DataFrame({"variable": x_cols_final}).to_csv(f"{self._output_path}/final_variables.csv", index=False)
        return df_data[x_cols_final + [y_col]]

    def preprocess_dataset(self, df_data, x_cols, y_col):
        df_descriptive_statistics = self.preprocess_descriptive_statistics(df_data, x_cols)
        df_data_preprocessed = df_data[x_cols + [y_col]]
        df_data_preprocessed = self.preprocess_impute_missing(df_data_preprocessed, x_cols)
        df_bivariate_analysis = self.preprocess_compute_bivariate_analysis(df_data_preprocessed, x_cols, y_col)
        df_corr_pairs_abs_cutoff = self.preprocess_compute_correlation_pairs(df_data_preprocessed, x_cols)
        df_data_preprocessed_clean = self.preprocess_clean_correlations(df_data_preprocessed, x_cols, y_col, df_corr_pairs_abs_cutoff, df_bivariate_analysis)

        return df_data_preprocessed_clean

def process_preprocess_dataset(x_cols, y_col):
    if (os.getcwd().endswith('src')):
        os.chdir("..")
    df_data_train = pd.read_csv("data/out/application_data_train.csv")
    preprocess_data_instance = preprocess_data("outputs")
    df_data_train_prepared = preprocess_data_instance.preprocess_dataset(df_data_train, x_cols, y_col)
    df_data_train_prepared.to_csv("data/out/application_data_train_prepared.csv", index=False)

def main():
    x_cols = ["CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_BIRTH","DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "OWN_CAR_AGE"]
    y_col = "TARGET"
    process_preprocess_dataset(x_cols, y_col)

if __name__ == "__main__":
    fire.Fire(main)
