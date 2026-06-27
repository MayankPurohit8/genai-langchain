from langchain_classic.document_loaders import TextLoader

loader = TextLoader('./9DocumentLoaders/text.txt',encoding = 'utf-8')

docs = loader.load()

print(docs[0])