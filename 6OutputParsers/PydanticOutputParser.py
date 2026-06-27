from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
import os
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task= "text-generation"
)

model = ChatHuggingFace(llm = llm)

class Person(BaseModel) : 
    name : str = Field(description='name of the person')
    age : int = Field(description='age of person' , gt=18)
    city : str = Field(description="City the person lives")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template= "give me name age and city of fictional person living in {country} \n {format_instruction}",
    input_variables= ['country'],
    partial_variables= {'format_instruction': parser.get_format_instructions()}
)

prompt =template.invoke({"country" : "Nepal"})


chain  = template | model | parser
res = chain.invoke({'country': 'Nepal'})
print(res)