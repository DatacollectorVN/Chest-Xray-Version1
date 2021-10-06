# CHEST XRAY VERSION 1 (Old version)
The model detect the abnormalities in chest-Xray image by using RetinaNet by Kos Nhan 

This project is currently version 1 by using transfer learning from [fizyr](https://github.com/fizyr/keras-retinanet)

## INTRODUCTION
This project used the dataset form [VinBigData](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection) to classify and localize 14 diseases in chest-Xray.

But after experiments, we decided to choose 5/14 diseases from that dataset (the reason, we have described in detail in Data_Preprocessing). 

Our result after 85 epochs:

| Disease | Aortic enlargement | Cardiomegaly | ILD | Pleural thickening | Pulmonary fibrosis |
| :---: | :---: | :---: | :---: | :---: | :---: |
| AP | 0.9751 | 0.9478 | 0.9478 | 0.6561 | 0.7104 |

mAP (Training): 0.8044 for 5 diseases

## INSTALLATION 
1. Create virtual environment
```bash
conda create -n myenv python=3.8
conda activate myenv
```
2. clone this repository 
3. Install required packages 
```bash 
pip install Keras_retinanet/.
pip install -r Keras_retinanet/requirements.txt
```
4. In the repository, execute `bash setup_data.sh` for create folder and download small dataset.

5. Download pretrain model 
```bash 
python config/download_model.py --dest Keras_retinanet/snapshots/pretrain_model.h5
```
6. Setup  
```bash 
cd Keras_retinanet
python setup.py build_ext --inplace
```
# FOR TRAINING 
7. Convert dataset to standard format
```bash 
python config/convert_data.py --Dataset_small/dataset_after_processing_small.csv --dest Keras_retinanet
```
8. Change directory in the Keras_retinanet folder and training
```bash 
cd Keras_retinanet
python keras_retinanet/bin/train.py --freeze-backbone  --workers 0 --weights snapshots/pretrain_model.h5 --backbone "resnet101"  --lr 0.00002  --batch-size 6 --steps 20  --image-min-side 900 --image-max-side 900 --epochs 2 csv annotation_5_classes.csv  classes_5.csv --val-annotations annotation_5_classes.csv
```

## FOR DEPLOYING STREAMLIT WEB APPLICATION
7. Run streamlit
```bash
streamlit run streamlit.py
```
Start to enjoy 
