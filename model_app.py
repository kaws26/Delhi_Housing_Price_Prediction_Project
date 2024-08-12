import logging.config
import sys
from src.DelhiHousingProject.logger import logging
from src.DelhiHousingProject.components.data_ingestion import DataIngestion
from src.DelhiHousingProject.components.data_transformation import DataTransformation
from src.DelhiHousingProject.components.model_training import ModelTrain
from src.DelhiHousingProject.exception import CustomException
import numpy as np

if __name__=="__main__":
    logging.info("The execution has started!")
    try:
        # data ingestion
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
        
        #data transformation
        data_transform=DataTransformation()
        train_arr,test_arr,_=data_transform.initiate_data_transformation(train_path=train_data_path,test_path=test_data_path)
        
        #model training
        model_train=ModelTrain()
        r2score=model_train.initiate_model_trainer(train_arr=train_arr,test_arr=test_arr)
        print(r2score)
        
    except Exception as e:
        raise CustomException(e,sys)