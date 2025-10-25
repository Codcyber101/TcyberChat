<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# AGENTS.md

A specification file to guide coding agents (or AI assistants) working on the Local RAG Chatbot project.

---

## Project Overview

- **Name**: Local First Chatbot
- **Purpose**: A comprehensive multi-modal chatbot API with local AI processing. Users upload documents (PDF, TXT, images, audio), ask questions via conversational AI, get RAG-enhanced answers with citations; supports image analysis, audio transcription, vector search, and data portability.
- **Stack**:
  - Frontend: React 18 + TypeScript + Vite + Tailwind CSS + Radix UI
  - Backend: Python 3.11 + FastAPI + SQLAlchemy + ChromaDB
  - AI/ML: LangChain + sentence-transformers + transformers + torch + OpenAI Whisper
  - Multi-modal Processing: Image analysis (OCR), audio transcription
  - Vector Store: ChromaDB (for local development)
  - LLM: Ollama (local) by default; optional hosted LLMs as needed

---

... (file truncated for brevity in remote snapshot)
