import tempfile, os
from langchain.document_loaders import TextLoader, PDFMinerLoader, UnstructuredWordDocumentLoader, MarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from .models import Document, RagSystem, RagQueryLog

LOADER_MAP = {
    'txt': TextLoader,
    'pdf': PDFMinerLoader,
    'docx': UnstructuredWordDocumentLoader,
    'md': MarkdownLoader,
}

class RagAssistant:
    def __init__(self, system: RagSystem):
        self.system = system
        self.embedding = OpenAIEmbeddings()
        self.index = self._load_index()

    def _load_index(self):
        config = self.system.index_config or {}
        if self.system.index_type == 'faiss':
            path = config.get('index_path') or f'/tmp/{self.system.name}.faiss'
            if os.path.exists(path):
                return FAISS.load_local(path, self.embedding)
            return FAISS(self.embedding.embed_query, [])
        # добавить другие индексаторы
        raise NotImplementedError

    def index_document(self, doc: Document):
        loader_cls = LOADER_MAP.get(doc.format)
        loader = loader_cls(file_path=doc.metadata.get('file_path') or tmp_file_for(doc))
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        self.index.add_documents(chunks)
        self.index.save_local(f'/tmp/{self.system.name}.faiss')

    def answer(self, query: str):
        qa = RetrievalQA.from_chain_type(
            llm=self.system.model_name,
            chain_type='stuff',
            retriever=self.index.as_retriever()
        )
        return qa.run(query)

def tmp_file_for(doc: Document):
    """Вспомогательная запись контента во временный файл."""
    suffix = f".{doc.format}"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(doc.content)
    return path
