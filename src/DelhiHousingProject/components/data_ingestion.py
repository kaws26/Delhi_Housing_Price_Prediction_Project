import os
import sys
import pandas as pd
from src.DelhiHousingProject.logger import logging
from src.DelhiHousingProject.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')
    
class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        try:
            #reading the data
            logging.info("Reading data from csv file")
            df=pd.read_csv('/Notebooks/Delhi_housing_data.csv')
            df=df.drop('Address',axis=1)
            df.drop('Status',inplace=True,axis=1)
            df.drop('Landmarks',inplace=True,axis=1)
            df.drop('desc',inplace=True,axis=1)
            df.drop('Price_sqft',inplace=True,axis=1)
            df.drop('Unnamed: 0',inplace=True,axis=1)
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)
            
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("Data Ingestion has been completed! ")
            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
            
            
            
        except Exception as e:
            raise CustomException(e,sys)
    