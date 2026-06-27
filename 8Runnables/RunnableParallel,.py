from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableParallel
load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task= "text-generation"
)

llm2 = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-14B-Instruct",
    task="text-generation"
)
model1 = ChatHuggingFace(llm = llm1)
model2 = ChatHuggingFace(llm = llm1)

prompt1 = PromptTemplate(
    template="Write short notes on the following text \n {text}",
    input_variables=['text']
)
prompt2 = PromptTemplate(
    template="generate 5 small questions on the following \n {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n {notes} \n {quiz}",
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

parallel_chain =RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser,
})

merge_chain = prompt3 | model1 | parser

chain  = parallel_chain | merge_chain

text ="""Support Vector Machines (SVM) are supervised machine learning algorithms primarily used for classification and regression tasks.  They work by finding the optimal hyperplane in an N-dimensional space that maximizes the margin between different classes, ensuring better generalization on unseen data. 

Key components and mechanisms include:

Support Vectors: The specific data points closest to the hyperplane that define the decision boundary and margin. 
Kernel Trick: A technique that maps non-linearly separable data into higher-dimensional spaces, allowing the SVM to find a linear separation where none existed in the original space. 
Margin Maximization: The algorithm prioritizes the widest possible gap between classes, which reduces the risk of overfitting and improves robustness against noise. 
SVMs are particularly effective for high-dimensional data and smaller datasets, with common applications in text categorization, image recognition, and bioinformatics.  While powerful, they can be computationally expensive for very large datasets and do not directly provide probability estimates. """
res = chain.invoke({'text' :text })

print(res)

chain.get_graph().print_ascii()