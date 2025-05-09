
### PDF
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader

```
Agglomerate text boxes into lines, paragraphs, and other structures via heuristics or ML inference;
Run OCR on images to detect text therein;
Classify text as belonging to paragraphs, lists, tables, or other structures;
Structure text into table rows and columns, or key-value pairs.

整合了众多 PDF parser
#### Text Extraction
简单识别正文中的文本字符

`pip install pypdf`
#### Layout Analysis

`pip install langchain-unstructured`

```python
loader = UnstructuredLoader(
    file_path=file_path,
    strategy="hi_res",
    partition_via_api=True,
    coordinates=True,
)
docs = []
for doc in loader.lazy_load():
    docs.append(doc)
```

#### Image OCR

### Web Pages
### HTML
### Markdown
### Office Data
### CSV
### JSON
### Customized Data
