`pip install longchain`


- https://python.langchain.com/docs/how_to/

### 数据处理
#### [Document Loader](document_loader.md)
1. PDF files
2. web pages
3. CSV data
4. data from a directory
5. HTML data
6. JSON data
7. Markdown data
8. Microsoft Office Data
9. customized document loader
#### Text Spliter

### Embedding相关
```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

vector_store = InMemoryVectorStore.from_documents(pages, OpenAIEmbeddings())
docs = vector_store.similarity_search("What is LayoutParser?", k=2)
for doc in docs:
    print(f'Page {doc.metadata["page"]}: {doc.page_content[:300]}\n')
```