import sys
import os
import pandas as pd
from src.DelhiHousingProject.exception import CustomException
from src.DelhiHousingProject.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            preprocessor=load_object(file_path=preprocessor_path)
            model=load_object(file_path=model_path)
            
            scaled_data=preprocessor.transform(features)
            preds=model.predict(scaled_data)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 area:float,
                 latitude:float,
                 longitude:float,
                 Bedrooms:int,
                 Bathrooms:int,
                 Balcony:int,
                 neworold:object,
                 parking:int,
                 Furnished_status:object,
                 Lift:int,
                 type_of_building:object):
        self.area=area
        self.latitude=latitude
        self.longitude=longitude
        self.Bedrooms=Bedrooms
        self.Bathrooms=Bathrooms
        self.Balcony=Balcony
        self.neworold=neworold
        self.parking=parking
        self.Furnished_status=Furnished_status
        self.Lift=Lift
        self.type_of_building=type_of_building
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "area":[self.area],
                "latitude":[self.latitude],
                "longitude":[self.longitude],
                "Bedrooms":[self.Bedrooms],
                "Bathrooms":[self.Bathrooms],
                "Balcony":[self.Balcony],
                "neworold":[self.neworold],
                "parking":[self.parking],
                "Furnished_status":[self.Furnished_status],
                "Lift":[self.Lift],
                "type_of_building":[self.type_of_building]
            }        
            
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)