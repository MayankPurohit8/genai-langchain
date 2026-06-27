#gives the input as ouput 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableParallel,RunnablePassthrough
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task= "text-generation"
)

model = ChatHuggingFace(llm = llm)

parser =StrOutputParser()

prompt1 = PromptTemplate(
    template = "Write a joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = "Write the explaination of following joke\n{text}",
    input_variables=['text']
)

joke_gen_chain = prompt1 | model | parser

parallel_chain = RunnableParallel(
    {
        'joke':RunnablePassthrough(),
        'explaination': prompt2 | model | parser
    }
)

final_chain = joke_gen_chain | parallel_chain

res = final_chain.invoke({'topic' : 'cricket'})
print(res)