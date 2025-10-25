# Backend

This folder contains the backend FastAPI application and tests for the Local First Chatbot.

Key notes
- Migrations: Alembic is used and a baseline migration is included. To run migrations locally use the project's Python environment and `alembic` from the `backend` folder.
- PDF extraction: We prefer PyMuPDF (`pymupdf`) for robust and fast PDF text extraction; the code falls back to `pypdf`/`PyPDF2` if PyMuPDF is not available.
- Tests: There are unit tests (`tests/unit`) and contract/integration tests (`tests/contract`, `tests/integration`).

Run tests locally

1. Activate the project's virtualenv (see project root README for `uv` steps) or:

```pwsh
cd backend
.venv\Scripts\activate
pip install -r requirements.txt
pytest tests/unit
pytest tests/contract
pytest tests/integration
```

CI
- A GitHub Actions workflow has been added at `.github/workflows/ci.yml` to run the backend tests on push / PR.
