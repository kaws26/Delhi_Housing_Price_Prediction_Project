import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os
from src.DelhiHousingProject.logger import logging
from src.DelhiHousingProject.exception import CustomException
from src.DelhiHousingProject.utils import save_object

@dataclass
class DataTransformationConfig:
        preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
        
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformer_obj(self):
        try:
            numerical_columns=['area','longitude','latitude','Bedrooms','Bathrooms','Balcony','parking','Lift']
            categorical_columns=[
                'neworold',
                'Furnished_status',
                'type_of_building'
                ]
            num_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(missing_values=np.nan,strategy='constant',fill_value=0)),
                ('scaler',StandardScaler())
                ])
            cat_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(missing_values=np.nan,strategy='most_frequent')),
                ('one_hot_encoder',OneHotEncoder(handle_unknown='ignore')),
                ('scaler',StandardScaler(with_mean=False))
                ])
            
            logging.info("Numerical columns: {}".format(numerical_columns))
            logging.info("Categorical columns: {}".format(categorical_columns))
            
            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)
            ])
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try: 
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("reading the training and testing file")
            preprocessor_obj=self.get_data_transformer_obj()
            target_column='price'
            
            #Splitting the train data into dependent and independent variable
            input_train_features=train_df.drop(target_column,axis=1)
            target_train_features=train_df[target_column]
            
            #splitting the testing data into dependent and independent
            input_test_features=test_df.drop(target_column,axis=1)
            target_test_features=test_df[target_column]
            
            logging.info("Applying preprocessing to train and test data")
            
            input_train_feature_arr=preprocessor_obj.fit_transform(input_train_features)
            input_test_feature_arr=preprocessor_obj.transform(input_test_features)
            
            train_arr=np.c_[input_train_feature_arr,np.array(target_train_features)]
            test_arr=np.c_[input_test_feature_arr,np.array(target_test_features)] 
            
            logging.info("saved preprocessing object")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
                )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
            
            
            
            
        except Exception as e:
            raise CustomException(e,sys)
        
        