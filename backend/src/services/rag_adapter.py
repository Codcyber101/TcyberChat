"""RAG adapter to centralize LangChain usage and provide safe stubs for tests.

This module exposes a small API that the rest of the app can use. When
LangChain is not installed or is incompatible, lightweight stubs are provided
to keep runtime stable for tests.
"""
from typing import Any, List, Optional

try:
    # Try to import selected LangChain utilities we use
    from langchain.memory import ConversationBufferWindowMemory  # type: ignore
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
    LANGCHAIN_PRESENT = True
except Exception:
    LANGCHAIN_PRESENT = False


class AdapterMemory:
    def __init__(self, k: int = 5):
        # If LangChain present, wrap its ConversationBufferWindowMemory to expose chat_memory
        if LANGCHAIN_PRESENT:
            try:
                self.memory = ConversationBufferWindowMemory(k=k, return_messages=True, memory_key="chat_memory")
                # Ensure chat_memory attribute exists
                self.chat_memory = self.memory.chat_memory
            except Exception:
                self.memory = None
                self.chat_memory = _ChatMemoryStub()
        else:
            self.memory = None
            self.chat_memory = _ChatMemoryStub()

    # Backwards-compatible methods
    def add_user(self, content: str):
        try:
            if hasattr(self.chat_memory, 'add_user_message'):
                self.chat_memory.add_user_message(content)
            elif hasattr(self.chat_memory, 'add_message'):
                self.chat_memory.add_message({'type': 'human', 'content': content})
        except Exception:
            pass

    def add_ai(self, content: str):
        try:
            if hasattr(self.chat_memory, 'add_ai_message'):
                self.chat_memory.add_ai_message(content)
            elif hasattr(self.chat_memory, 'add_message'):
                self.chat_memory.add_message({'type': 'ai', 'content': content})
        except Exception:
            pass

    def get_context(self) -> List[dict]:
        msgs = getattr(self.chat_memory, 'messages', [])
        out = []
        for m in msgs:
            try:
                # Normalize supported shapes
                if isinstance(m, dict):
                    role = 'user' if m.get('type') in ('human', 'user') else 'assistant'
                    content = m.get('content', '')
                else:
                    role = 'user' if m.__class__.__name__.lower().startswith('human') else 'assistant'
                    content = getattr(m, 'content', str(m))
                out.append({'role': role, 'content': content})
            except Exception:
                continue
        return out


class AdapterSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if LANGCHAIN_PRESENT:
            self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            self.splitter = None

    def split(self, text: str) -> List[str]:
        if self.splitter:
            try:
                return self.splitter.split_text(text)
            except Exception:
                return [text]
        return [text]



class _ChatMemoryStub:
    """A small in-memory chat memory stub exposing the interface expected by the app.

    Methods:
    - add_message(dict|object)
    - add_user_message(str)
    - add_ai_message(str)
    - messages (list)
    - clear()
    """

    def __init__(self):
        self.messages = []

    def add_message(self, message):
        # normalize both dict and simple objects
        if isinstance(message, dict):
            self.messages.append(message)
        else:
            try:
                content = getattr(message, 'content', str(message))
                mtype = getattr(message, 'type', None) or getattr(message, '__class__', type(message)).__name__
                self.messages.append({'type': mtype, 'content': content})
            except Exception:
                self.messages.append({'type': 'unknown', 'content': str(message)})

    def add_user_message(self, content: str):
        self.messages.append({'type': 'human', 'content': content})

    def add_ai_message(self, content: str):
        self.messages.append({'type': 'ai', 'content': content})

    def clear(self):
        self.messages = []


def create_memory(k: int = 5) -> AdapterMemory:
    return AdapterMemory(k=k)


def create_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> AdapterSplitter:
    return AdapterSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def create_embeddings(model_name: str = "all-MiniLM-L6-v2") -> Optional[Any]:
    """Return an embeddings object or None when LangChain embeddings not present."""
    if LANGCHAIN_PRESENT:
        try:
            from langchain_community.embeddings import SentenceTransformerEmbeddings

            return SentenceTransformerEmbeddings(model_name=model_name)
        except Exception:
            return None
    return None


def create_vectorstore(client: Any, collection_name: str = "documents", embedding=None) -> Optional[Any]:
    """Return a vectorstore instance or None when not available."""
    if LANGCHAIN_PRESENT:
        try:
            from langchain_community.vectorstores import Chroma

            return Chroma(client=client, collection_name=collection_name, embedding_function=embedding)
        except Exception:
            return None
    return None


class AIServiceLLMAdapter:
    """Minimal adapter that exposes a .generate(prompt) method backed by the AI service."""

    def __init__(self, ai_service):
        self.ai_service = ai_service

    def generate(self, prompt: str) -> str:
        # Support sync or async AI service
        try:
            if hasattr(self.ai_service, 'generate_response'):
                # ai_service.generate_response may be async; try sync first
                result = self.ai_service.generate_response(prompt=prompt)
                # If coroutine, run it
                if hasattr(result, '__await__'):
                    import asyncio

                    return asyncio.get_event_loop().run_until_complete(result)
                return result
        except Exception:
            pass
        # Last resort: str() the service
        return str(self.ai_service)
