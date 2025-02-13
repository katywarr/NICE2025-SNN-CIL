from pathlib import Path
import pandas as pd
import os

def combine_dicts(dicts: list) -> dict:
    all_dicts = {}
    for d in dicts:
        all_dicts = all_dicts | d
    return all_dicts


def pandas_from_dicts(dicts: list) -> pd.DataFrame:
    all_dicts = combine_dicts(dicts)
    return pd.DataFrame([all_dicts])


class ExcelHandler:

    def __init__(self, file_dir: str, file_name: str):
        """
        :param file_dir:
        Directory of the file to write into.
        :param file_name:
        Name of the file.
        """
        file_dir_path = Path(file_dir)
        self.full_file_name = file_dir+os.sep+file_name+'.xlsx'
        print('****** FUll file path for excel ', self.full_file_name)
        self.file_path = None
        if not file_dir_path.is_dir():
            print('Error: Directory {} passed to ExcelWriter does not exist or is not a directory.\n'
                  '       No data will be written to {}. Current directory is {}'
                  .format(file_dir, self.full_file_name, os.getcwd()))
        else:
            self.file_path = Path(self.full_file_name)

    def add_sheet(self, df: pd.DataFrame, sheet_name: str):
        if self.file_path is None:
            self.__write_error(df, sheet_name)
            return

        if self.file_path.is_file():
            # Add new sheet to file
            with pd.ExcelWriter(self.full_file_name, mode='a', if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            # Create new file and add sheet
            df.to_excel(self.file_path, sheet_name=sheet_name, index=False)
        return

    def read_sheet(self, sheet_name: str) -> pd.DataFrame:
        df = pd.DataFrame()
        # print('ExcelHandler: Reading data from file: {} sheet: {}'.format(self.file_path, sheet_name))
        if self.file_path is None or not self.file_path.is_file():
            self.__read_error_sheet(sheet_name)
        else:
            df = pd.read_excel(self.full_file_name, sheet_name=sheet_name)
        return df

    def check_sheet_name(self, sheet_name: str) -> bool:
        ret_val = False
        if self.file_path is None or not self.file_path.is_file():
            self.__read_error()
        else:
            sheet_names = pd.ExcelFile(self.full_file_name).sheet_names
            if sheet_name in sheet_names:
                ret_val = True
        return ret_val

    def add_rows(self, df: pd.DataFrame, sheet_name: str):
        if self.file_path is None:
            self.__write_error(df, sheet_name)
            return False
        if self.file_path.is_file():
            try:
                pd.read_excel(self.full_file_name, sheet_name=sheet_name)
                # Sheet exists - append data
                with pd.ExcelWriter(self.full_file_name, mode="a", engine="openpyxl",
                                    if_sheet_exists="overlay") as writer:
                    start_row = writer.book[sheet_name].max_row
                    df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row, header=False)
            except ValueError:
                # Create a new sheet
                self.add_sheet(df=df, sheet_name=sheet_name)
        else:
            # Call add_sheet which will create the file and add a fresh sheet for the row
            self.add_sheet(df=df, sheet_name=sheet_name)

    def add_row_from_dicts(self, dicts: list, sheet_name: str):
        df = pandas_from_dicts(dicts=dicts)
        self.add_rows(df=df, sheet_name=sheet_name)

    def __write_error(self, df: pd.DataFrame, sheet_name: str):
        print('Error: Unable to write data to sheet {} as the ExcelWriter was not successfully initialised.\n'
              '       This is probably because the directory for the target file does not exist '
              '(see previous messages).\n'
              '       The following data frame would have been written to the file:\n {}'
              .format(sheet_name, df, self.full_file_name))

    def __read_error(self):
        msg = '\nUnable to read data from file: ' + self.full_file_name + \
              '\nCurrent working directory is:' + os.getcwd()
        raise ValueError(msg)

    def __read_error_sheet(self, sheet_name: str):
        msg = '\nUnable to read data from sheet: ' + sheet_name + ' in  file: ' + self.full_file_name + \
              '\nCurrent working directory is:' + os.getcwd()
        raise ValueError(msg)




