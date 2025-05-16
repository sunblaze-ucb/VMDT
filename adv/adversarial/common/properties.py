from typing import Optional
from pydantic import BaseModel

class BaseProperty(BaseModel):
    pass

class ActionRecognitionProperty(BaseProperty):
    obj: str
    action: str
    target: Optional[str]= None
    
    def _get_core_property(self):
        return self.action
    
class AttributeRecognitionProperty(BaseProperty):
    obj: str
    attribute: str
    
    def _get_core_property(self):
        return self.attribute

class CountingProperty(BaseProperty):
    obj: str
    count: str
    
    def _get_core_property(self):
        return self.count
    
class ObjectRecognitionProperty(BaseProperty):
    obj: str
    
    def _get_core_property(self):
        return self.obj
    
class SpatialUnderstandingProperty(BaseProperty):
    obj_1: str
    relation: str
    obj_2: str
    
    def _get_core_property(self):
        return self.relation