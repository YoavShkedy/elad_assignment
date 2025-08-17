# LangGraph RAG Chatbot Assignment

## Objective

Build a chatbot that conducts a conversation on a defined topic (X) and integrates information from an external knowledge base using RAG (Retrieval-Augmented Generation) with LangGraph.

## Execution Requirements

### Data Collection & Preparation

- Select a domain/topic (X) of your choice (e.g., Cyber Security, Smart Agriculture, etc.)
- Collect a minimum of 10 relevant documents (PDF, HTML, Markdown)
- Perform preprocessing: cleaning and splitting into chunks

### Indexing

- Create embeddings (using OpenAI / Cohere / HuggingFace)
- Load the embeddings into a search engine (Vector DB) – e.g., Pinecone, Weaviate, FAISS, Chroma
- Provide a technological justification for your choice of tools

### Building the LangGraph Agent

Create a graph that includes:
- Node for receiving the question
- Node for searching the search engine
- Node for processing the retrieved information with GPT/Claude
- (Optional) Memory for maintaining context between conversation stages
- The architecture should be modular and ready for expansion

### Answer Retrieval & Source Display

- Construct answers that incorporate retrieved information
- Cite the source(s) in the response
- (Bonus) Add automatic follow-up questions and expand the conversation context

### Basic User Interface

- A UI is required to use the solution
- Candidate may choose the implementation technology (React, Angular, Streamlit, or other)
- The UI should allow the user to input a question and receive an answer
- No complex design is required – priority is functionality and ease of use

## Submission

- **Completed project** – preferably as a GitHub link with clean, modular code. A ZIP file is also acceptable
- **README including:**
  - Architecture description
  - Chosen technologies and explanations
  - Installation and run instructions
  - Short demo video showcasing the system

## Success Criteria

- **Architecture:** System-level thinking, modular design
- **LangGraph Usage:** Proper use of state machine, transition management
- **RAG Capabilities:** Quality of integration between retrieval and response
- **Code Quality:** Structure, readability, documentation
- **UI:** Functionality and ease of use
- **Product Thinking:** Awareness of RAG limitations and creative solutions

## Extra Points

- Use of streaming responses and/or session memory management
