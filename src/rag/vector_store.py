from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rag.config import EMBEDDING_MODEL

class VectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = None
        self.schema_metadata = {}
        
    def build_from_schemas(self, schemas: Dict[str, Any]):
        documents = []
        for schema_name, schema_info in schemas.items():
            metadata = schema_info['metadata']
            text_content = self._create_text_representation(schema_name, metadata)
            
            doc = Document(
                page_content=text_content,
                metadata={
                    'schema_name': schema_name,
                    'filename': schema_info['filename']
                }
            )
            documents.append(doc)
            self.schema_metadata[schema_name] = metadata
        
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
    def _create_text_representation(self, schema_name: str, metadata: Dict) -> str:
        text_parts = [f"Schema: {schema_name}"]
        
        if metadata.get('description'):
            text_parts.append(f"Description: {metadata['description']}")
        
        if metadata.get('properties'):
            text_parts.append("Properties:")
            for prop_name, prop_info in metadata['properties'].items():
                prop_text = f"  {prop_name}: {prop_info.get('description', '')}"
                if prop_info.get('type'):
                    prop_text += f" (Type: {prop_info['type']})"
                text_parts.append(prop_text)
        
        if metadata.get('definitions'):
            text_parts.append("Definitions:")
            for def_name, def_info in metadata['definitions'].items():
                def_text = f"  {def_name}: {def_info.get('description', '')}"
                text_parts.append(def_text)
                if def_info.get('properties'):
                    for prop_name, prop_info in def_info['properties'].items():
                        text_parts.append(f"    - {prop_name}: {prop_info.get('description', '')}")
        
        return "\n".join(text_parts)
    
    def retrieve_top_k(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call build_from_schemas first.")
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        retrieved_schemas = []
        for doc, score in results:
            schema_name = doc.metadata['schema_name']
            retrieved_schemas.append({
                'schema_name': schema_name,
                'filename': doc.metadata['filename'],
                'metadata': self.schema_metadata.get(schema_name, {}),
                'score': float(score),
                'content': doc.page_content
            })
        
        return retrieved_schemas