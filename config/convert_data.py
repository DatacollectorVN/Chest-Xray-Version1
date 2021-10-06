import os 
import pandas as pd 
import numpy as np
#### RUN THIS FILE AFTER YOU HAVE DATASET_SMALL
def main():
    BASE_DIR_DATASET = os.path.join('Dataset_small')
    dataset_csv_file = os.path.join(BASE_DIR_DATASET, 'dataset_after_processing_small.csv')
    
    BASE_DIR_DATASET = os.path.join('..','Dataset_small')
    train_dir = os.path.join('train', 'train')
    train_dir = os.path.join(BASE_DIR_DATASET, train_dir)


    df =  pd.read_csv(dataset_csv_file, index_col = 0)
    df['image_path'] = df.image_id.map(lambda x: os.path.join(train_dir, x + '.jpg'))
    csv_file_save = 'Keras_retinanet'
    csv_file_save = os.path.join(csv_file_save, 'annotation_5_classes.csv') 

    ### SAVE ANNOTATION FILE 
    df_new = df[['image_path', 'x_min', 'y_min', 'x_max', 'y_max', 'class_name']]
    df_new[['x_min', 'y_min', 'x_max', 'y_max']]= df_new[['x_min', 'y_min', 'x_max', 'y_max']].astype(int)
    df_new.to_csv(csv_file_save, header = None, index = None)

    ### SAVE CLASSES FILE
    df_new = pd.DataFrame({'class_name': ['Aortic enlargement', 'Cardiomegaly', 'ILD', 'Pleural thickening',
                                          'Pulmonary fibrosis'], 
                       'class_id': [0, 1, 2, 3, 4] })
    
    csv_file_save = 'Keras_retinanet'
    csv_file_save = os.path.join(csv_file_save, 'classes_5.csv')
    df_new.to_csv(csv_file_save, header = None, index = None)

if __name__ == '__main__':
    main()
