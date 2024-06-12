import pandas as pd
import numpy as np
from glob import glob
from os.path import splitext, basename, exists
from sklearn.preprocessing import minmax_scale
import xlrd
from openpyxl.utils.cell import coordinate_to_tuple

def extract_file_paths(ground_truth_folder, tasks_folder, tasks_dict, selected_tasks):

    count = 0

    dataset_df = pd.DataFrame(columns=['Spirometry_file'] + [f'{task_name}_file' for task_name in selected_tasks] )

    skipped = 0

    for spiro_file in glob(f'{ground_truth_folder}/*.xls'):
        #Getting the file name to get the groundtruth. Eg. getting '1' from 'folder/1.wav'
        recording_index = (splitext(basename(spiro_file))[0]).split('_')[0]
        
        task_files = []

        for task_name in selected_tasks:
          task = tasks_dict[task_name]
          task_folder = task['folder']
          task_suffix = task['suffix']
          task_file = f'{tasks_folder}/{task_folder}/{recording_index}_{task_suffix}.wav'

          if not (exists(task_file)):
            skipped = skipped + 1
            task_file = None
          task_files.append(task_file)

        dataset_df.loc[len(dataset_df.index)] = [spiro_file] + task_files
        count = count + 1

    print(f'{count} spirometry readings found and {skipped} recordings missing')

    return dataset_df

def get_target_columns(row, cell_coordinates):

    file_path = row['Spirometry_file']
    return read_cells_from_excel(file_path, cell_coordinates)

def read_cells_from_excel(filepath = None, cell_coordinates = None):

    wb = xlrd.open_workbook(filename = filepath)
    sheet = wb.sheet_by_index(0)
    values = []
    for coordinates in cell_coordinates:
        x, y = coordinate_to_tuple(coordinates)
        values.append(np.float64(str(sheet.cell_value(rowx=x-1, colx=y-1)).replace('*','')))
    return tuple(values)

ground_truth_folder = './spirometry_ground_truth'
tasks_folder =  './Trimmed Data from phone'

#This is a dictionary for location of all the target variables in the xls file
cell_locations = {'FVC': 'B23', 'FEV1': 'B24', 'FEV1/FVC': 'B25'}

tasks_dict = {
            'O_Single': {
                                'folder': '5_2_O_single', 
                                'suffix': 'vowelo'},
            'Deep_Breath': {
                                'folder': '9_3_deepbreath', 
                                'suffix': 'deepbreaths'}
            }  #This data has other modalities but we mainly used thoes two in our benchmar

tasks = tasks_dict.keys()
print(tasks)

dataset_df = extract_file_paths(ground_truth_folder, tasks_folder, tasks_dict, tasks)

def extract_targets(selected_options):  
    if len(selected_options) == 0:
        print('No targets selected, please choose an option')
    else:
        print(f'Extracting {selected_options} values from ground truth folder: {ground_truth_folder}...')
        cell_coordinates = [cell_locations[x] for x in selected_options]
        
        global target_names, dataset_df
        target_names = selected_options
        
        #Extracting target variables based on choice
        dataset_df[selected_options] = dataset_df.apply(get_target_columns, axis=1, args=(cell_coordinates,), result_type='expand')

        print(f'Targets extracted')
        
extract_targets(['FVC','FEV1','FEV1/FVC'])

dataset_df = dataset_df.sort_values(by='Spirometry_file')

print(dataset_df)

dataset_df.to_excel('All_path.xlsx')