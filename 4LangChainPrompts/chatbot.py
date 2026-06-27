from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.messages import SystemMessage , HumanMessage,AIMessage
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task= "text-generation"
)

messages = [
    SystemMessage(content= "You are a helpful assistant responds in short messages")
]

model = ChatHuggingFace(llm = llm)

while True:
    user_input = input("You : ")
    messages.append(HumanMessage(content=user_input))
    if user_input=='exit':
        break
    result = model.invoke(messages)
    messages.append(AIMessage(content=result.content))
    print("AI : ",result.content)