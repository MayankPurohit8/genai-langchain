from pydantic import BaseModel
from typing import  Optional
class Student(BaseModel):
    name : str = "Mayank" #default value
    age : Optional[int] = None 

new_student = {'age':30}
student = Student(**new_student)
print(student)