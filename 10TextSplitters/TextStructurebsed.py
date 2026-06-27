from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

text = '''If your goal is to become a GenAI Engineer, you don't need to master every deep learning architecture from scratch. A solid conceptual understanding—especially of Transformers, embeddings, and how modern LLMs work—combined with strong software engineering skills will take you much further than spending months implementing CNNs or RNNs from scratch.

Given your current trajectory, I'd spend about 70% of my learning time on AI engineering and GenAI tooling and 30% on deep learning fundamentals. That balance aligns well with what many companies hiring GenAI engineers are looking for today.'''

splitter  = RecursiveCharacterTextSplitter(
    chunk_size = 10,
    chunk_overlap = 0,
)
res = splitter.split_text(text)

print(res)