#gives the input as ouput 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch
from pydantic import BaseModel,Field
from langchain_core.output_parsers import PydanticOutputParser
from typing import Literal
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task= "text-generation"
)

model = ChatHuggingFace(llm = llm)

parser =StrOutputParser()

class Review_sentiment(BaseModel):
    sentiment : Literal["pos","neg"] = Field(description="Sentiment of review either pos or neg")

parser = PydanticOutputParser(pydantic_object=Review_sentiment)

prompt1 = PromptTemplate(
    template = "Write the sentiment of following review \n {review} \n {follow_instruction}",
    input_variables=['review'],
    partial_variables= { "follow_instruction": parser.get_format_instructions()}
)

sentiment = prompt1 | model | parser


prompt2 = PromptTemplate(
    template= "Write a sorry feedback for the following review \n {review}",
    input_variables=['review']
)
prompt3 = PromptTemplate(
    template= "Write a thanking feedback for the following review \n {review}",
    input_variables=['review']
)

strparser = StrOutputParser()
branch_chain = RunnableBranch(
    (lambda x : x.sentiment == "neg", prompt2 | model | strparser),
    prompt3 | model | strparser
)

main_chain = sentiment | branch_chain

res = main_chain.invoke({"review": "this smartphone is very bad"})

print(res)