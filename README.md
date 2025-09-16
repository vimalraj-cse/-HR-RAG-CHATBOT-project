HR RAG Chatbot 🤖📄

📌 Features



Document ingestion \& embedding (Sentence Transformers + FAISS)



Query processing with lightweight re-ranking (BM25)



REST API using FastAPI



1.Interactive UI with Streamlit



2.Dockerized frontend \& backend for easy deployment



3.This project is a Retrieval-Augmented Generation (RAG) based chatbot designed for answering queries from HR policy documents.

4.It has a FastAPI backend and a Streamlit frontend, both containerized with Docker.


A. Clone the repo



1.git clone https://github.com/your-username/hr-rag-  chatbot.git



2.cd hr-rag-chatbot



B. Backend Setup

cd backend

python -m venv .venv

.venv\\Scripts\\activate   # Windows

source .venv/bin/activate   # Linux/Mac



pip install --upgrade pip

pip install -r requirements.txt



c.Run backend:

uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload



D.3. Frontend Setup



cd ../frontend

python -m venv .venv

.venv\\Scripts\\activate   # Windows

source .venv/bin/activate   # Linux/Mac



pip install -r requirements.txt

F.Run frontend:

streamlit run streamlit\_app.py



G. Open in browser



Backend API → http://localhost:8000/docs



Frontend UI → http://localhost:8501



streamlit run streamlit\_app.py



4\. Open in browser



Backend API → http://localhost:8000/docs



Frontend UI → http://localhost:8501



🐳 Running with Docker

 1.Build image 

&nbsp;  

&nbsp;  docker compose build



&nbsp;2.Run Containers

&nbsp;  

&nbsp;   docker compose up

3. Open in browser



Backend → http://localhost:8000



Frontend → http://localhost:8501


📂 Project Structure

hr-rag-chatbot/

│── backend/

│   ├── app/

│   ├── requirements.txt

│   └── Dockerfile.backend

│── frontend/

│   ├── streamlit\_app.py

│   ├── requirements.txt

│   └── Dockerfile.frontend

│── docker-compose.yml

│── README.md




