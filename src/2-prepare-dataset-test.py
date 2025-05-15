import fire
import pandas as pd
import os


class prepare_data():
    _output_path = ""

    def __init__(self, output_path):
        self._output_path = output_path

    def prepare_impute_missing(self, df_data, x_cols):
        df_data_imputed = df_data.copy()
        df_impute_parameters = pd.read_csv(f"{self._output_path}/impute_missing_parameters.csv")
        for col in x_cols:
            impute_value = df_impute_parameters[df_impute_parameters["variable"]==col]["impute_value"]
            df_data_imputed[col] = df_data_imputed[col].fillna(impute_value)
        return df_data_imputed

    def prepare_dataset(self, df_data, y_col):
        x_cols = pd.read_csv(f"{self._output_path}/final_variables.csv")["variable"].values.tolist()
        df_data_prepared = df_data[x_cols + [y_col]]
        df_data_prepared = self.prepare_impute_missing(df_data_prepared, x_cols)

        return df_data_prepared

def process_prepare_dataset(y_col):
    if (os.getcwd().endswith('src')):
        os.chdir("..")
    df_data_test = pd.read_csv("data/out/application_data_test.csv")
    prepare_data_instance = prepare_data("outputs")
    df_data_test_prepared = prepare_data_instance.prepare_dataset(df_data_test, y_col)
    df_data_test_prepared.to_csv("data/out/application_data_test_prepared.csv", index=False)


def main():
    y_col = "TARGET"
    process_prepare_dataset(y_col)

if __name__ == "__main__":
    fire.Fire(main)
