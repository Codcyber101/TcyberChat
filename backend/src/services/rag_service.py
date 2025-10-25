"""
RAGService for retrieval-augmented generation using comprehensive LangChain features
"""

from typing import List, Dict, Optional, Any, Tuple, AsyncGenerator
import asyncio
from uuid import uuid4

# Use the adapter to centralize LangChain usage and provide stable fallbacks
from .rag_adapter import (
    create_memory,
    create_splitter,
    create_embeddings,
    create_vectorstore,
    AIServiceLLMAdapter,
    LANGCHAIN_PRESENT,
)

LANGCHAIN_AVAILABLE = LANGCHAIN_PRESENT

# Minimal stub used when LangChain pieces are not present.
class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_template(cls, *a, **k):
        return cls()

    @classmethod
    def from_messages(cls, *a, **k):
        return cls()

    @classmethod
    def from_documents(cls, docs, **k):
        return cls()

    def split_text(self, text: str):
        return [text]

    def add_documents(self, docs):
        return

    def add_texts(self, texts, metadatas=None, ids=None):
        return

    def persist(self):
        return

    def get(self):
        return {'documents': [], 'metadatas': [], 'ids': []}

    def as_retriever(self, **k):
        return self

    def get_relevant_documents(self, query, k=5):
        return []

    def count(self):
        return 0

# Map minimal names to stubs for the non-LangChain path. More detailed behavior
# is provided by the adapter when LangChain is present.
if not LANGCHAIN_AVAILABLE:
    Chroma = _Stub
    SentenceTransformerEmbeddings = _Stub
    LangChainDocument = _Stub
    BaseRetriever = _Stub
    BaseCallbackHandler = object
    StrOutputParser = _Stub
    RunnablePassthrough = _Stub
    ChatPromptTemplate = _Stub
    MessagesPlaceholder = _Stub
    PromptTemplate = _Stub
    HumanMessage = str
    AIMessage = str
    BaseChatModel = object
    BaseLLM = object
    Generation = dict
    LLMResult = dict
    create_stuff_documents_chain = lambda *a, **k: _Stub()
    create_history_aware_retriever = lambda *a, **k: _Stub()
    create_retrieval_chain = lambda *a, **k: _Stub()
    BM25Retriever = _Stub
    EnsembleRetriever = _Stub
    DocumentCompressorPipeline = _Stub
    EmbeddingsFilter = _Stub
    RecursiveCharacterTextSplitter = _Stub
    ConversationBufferWindowMemory = _Stub

from .ai_service import get_ai_service
from ..database import chroma_client


class RAGCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for RAG operations"""

    def __init__(self):
        self.operations = []

    def on_chain_start(self, serialized, inputs, **kwargs):
        self.operations.append({
            "operation": "chain_start",
            "chain": serialized.get("name", "unknown"),
            "timestamp": asyncio.get_event_loop().time()
        })

    def on_chain_end(self, outputs, **kwargs):
        self.operations.append({
            "operation": "chain_end",
            "timestamp": asyncio.get_event_loop().time()
        })

    def on_retriever_start(self, serialized, query, **kwargs):
        self.operations.append({
            "operation": "retrieval_start",
            "query": query[:100] + "..." if len(query) > 100 else query
        })

    def on_retriever_end(self, documents, **kwargs):
        self.operations.append({
            "operation": "retrieval_end",
            "documents_found": len(documents)
        })


class RAGService:
    """Service for retrieval-augmented generation using comprehensive LangChain features"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.embeddings = None
        self.text_splitter = None
        self.conversation_memory = None
        self.callback_handler = None
        self.ai_service = get_ai_service()

        # Advanced retrievers and chains
        self.ensemble_retriever = None
        self.conversational_chain = None
        self.rag_chain = None
        self.compression_retriever = None

        if LANGCHAIN_AVAILABLE:
            self._initialize_langchain_components()
        else:
            # Use adapter fallbacks when LangChain is not available
            from .rag_adapter import create_memory, create_splitter

            self.conversation_memory = create_memory(k=5)
            self.text_splitter = create_splitter(chunk_size=1000, chunk_overlap=200)

    def _initialize_langchain_components(self):
        """Initialize comprehensive LangChain components for RAG"""
        try:
            # Initialize embeddings, vectorstore, splitter and memory through the adapter
            # Adapter will try to use LangChain components when available, otherwise
            # return safe None/stubs so tests and CI remain stable.
            self.embeddings = create_embeddings(model_name="all-MiniLM-L6-v2")

            # Use shared ChromaDB client from database module
            self.vectorstore = create_vectorstore(client=chroma_client, collection_name="documents", embedding=self.embeddings)

            # Initialize text splitter for document chunking
            self.text_splitter = create_splitter(chunk_size=1000, chunk_overlap=200)

            # Initialize conversation memory
            self.conversation_memory = create_memory(k=5)

            # Initialize callback handler for observability
            self.callback_handler = RAGCallbackHandler()

            # Create advanced retrievers and chains if LangChain pieces are available
            self._setup_advanced_retrievers()
            self._setup_chains()

        except Exception as e:
            print(f"Failed to initialize LangChain components: {e}")
            self.vectorstore = None
            self.embeddings = None

    def _setup_advanced_retrievers(self):
        """Set up advanced retrieval strategies"""
        # If LangChain isn't available or vectorstore wasn't initialized, skip advanced retrievers
        if not LANGCHAIN_AVAILABLE or not self.vectorstore:
            return

        try:
            # Base retrievers
            similarity_retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}
            )

            mmr_retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 10, "lambda_mult": 0.5}
            )

            # BM25 retriever (keyword-based) - would need documents list
            # For now, create ensemble with available retrievers
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[similarity_retriever, mmr_retriever],
                weights=[0.7, 0.3]  # Weight similarity search higher
            )

            # Document compression pipeline
            embeddings_filter = EmbeddingsFilter(
                embeddings=self.embeddings,
                similarity_threshold=0.7
            )

            self.compression_retriever = DocumentCompressorPipeline(
                transformers=[embeddings_filter]
            )

        except Exception as e:
            print(f"Failed to setup advanced retrievers: {e}")

    def _setup_chains(self):
        """Set up comprehensive LangChain chains"""
        # If LangChain pieces aren't available, skip chain setup
        if not LANGCHAIN_AVAILABLE:
            return

        try:
            # Create custom LLM wrapper for our AI service
            # Wrap AI service in a minimal LLM-like adapter only if BaseLLM is available
            if isinstance(BaseLLM, type):
                class AIServiceLLM(BaseLLM):
                    # Declare fields so pydantic/BaseModel-style initialization accepts them
                    ai_service: Any
                    callback_handler: Optional[BaseCallbackHandler] = None

                    def __init__(self, ai_service, callback_handler=None):
                        # BaseLLM may be a simple class; avoid relying on pydantic behavior here
                        try:
                            super().__init__()
                        except Exception:
                            pass
                        self.ai_service = ai_service
                        self.callback_handler = callback_handler

                    @property
                    def _llm_type(self) -> str:
                        return "custom_ai_service"

                    async def _acall(self, prompt: str, **kwargs) -> str:
                        if self.callback_handler and hasattr(self.callback_handler, 'on_chain_start'):
                            try:
                                self.callback_handler.on_chain_start({'name': 'ai_service_call'}, {'prompt': prompt})
                            except TypeError:
                                # Some callback handlers require additional args; ignore for dev
                                pass

                        response = await self.ai_service.generate_response(prompt=prompt)

                        if self.callback_handler and hasattr(self.callback_handler, 'on_chain_end'):
                            try:
                                self.callback_handler.on_chain_end({'response': response})
                            except TypeError:
                                pass

                        # Ensure string return
                        if isinstance(response, dict):
                            return str(response.get('response', ''))
                        return str(response)

                    def _call(self, prompt: str, **kwargs) -> str:
                        # Synchronous fallback
                        return asyncio.get_event_loop().run_until_complete(self._acall(prompt, **kwargs))

                    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs: Any) -> Any:
                        generations = []
                        for prompt in prompts:
                            text = self._call(prompt, **kwargs)
                            if Generation is not None:
                                generations.append([Generation(text=text)])
                            else:
                                generations.append([{'text': text}])
                        if LLMResult is not None:
                            return LLMResult(generations=generations)
                        return {'generations': generations}

                    async def _agenerate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs: Any) -> Any:
                        generations = []
                        for prompt in prompts:
                            text = await self._acall(prompt, **kwargs)
                            if Generation is not None:
                                generations.append([Generation(text=text)])
                            else:
                                generations.append([{'text': text}])
                        if LLMResult is not None:
                            return LLMResult(generations=generations)
                        return {'generations': generations}

            custom_llm = AIServiceLLM(self.ai_service, self.callback_handler)

            # 1. Basic RAG Chain
            rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant that answers questions based on the provided context.
Use the following pieces of context to answer the question at the end.

Guidelines:
- If you don't know the answer based on the context, say so clearly
- Provide specific references to the source material when possible
- Be concise but comprehensive
- Use citations when referencing specific information

Context:
{context}

Question: {question}

Answer:""")

            # Create stuff documents chain
            stuff_chain = create_stuff_documents_chain(custom_llm, rag_prompt)

            # Create retrieval chain
            self.rag_chain = create_retrieval_chain(
                self.ensemble_retriever,
                stuff_chain
            )

            # 2. Conversational RAG Chain with memory
            condense_prompt = PromptTemplate.from_template("""
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone question:""")

            history_aware_retriever = create_history_aware_retriever(
                custom_llm, self.ensemble_retriever, condense_prompt
            )

            conversational_prompt = ChatPromptTemplate.from_messages([
                ("system", """
You are a helpful AI assistant with access to document knowledge. Use the provided context and conversation history to answer questions.

Guidelines:
- Use both the retrieved context and conversation history
- Maintain coherence across the conversation
- Cite sources when providing specific information
- Ask for clarification if needed

Context: {context}
Conversation History: {chat_history}
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])

            conversational_stuff_chain = create_stuff_documents_chain(
                custom_llm, conversational_prompt
            )

            self.conversational_chain = create_retrieval_chain(
                history_aware_retriever,
                conversational_stuff_chain
            )

        except Exception as e:
            print(f"Failed to setup chains: {e}")

    async def generate_rag_streaming_response(self, query: str, document_id: Optional[str] = None,
                                              max_context_chunks: int = 3, conversational: bool = False,
                                              chat_history: Optional[List[Dict[str, str]]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate RAG-enhanced streaming response using advanced LangChain chains"""
        full_response_content = ""
        citations = []
        try:
            if conversational and chat_history and self.conversational_chain:
                # Use conversational RAG chain with memory
                formatted_history = []
                if chat_history:
                    for msg in chat_history[-10:]:  # Keep last 10 messages
                        if msg.get("role") == "user":
                            formatted_history.append(HumanMessage(content=msg.get("content", "")))
                        elif msg.get("role") == "assistant":
                            formatted_history.append(AIMessage(content=msg.get("content", "")))

                        ... (truncated for brevity) ...