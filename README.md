# Conversational RAG Chatbot

### Overview

The **Conversational RAG Chatbot** is an AI-powered application designed to provide users with the ability to interact with PDF documents conversationally. By leveraging state-of-the-art AI models and a retrieval-augmented generation (RAG) approach, the chatbot enables users to upload PDFs and ask questions about the content in natural language.

### Key Features

1. **PDF Upload**: Users can upload one or more PDF files through the sidebar to interact with the content.
  
2. **Text Chunking**: The application processes the PDF documents using a text splitter, ensuring the content is split into manageable chunks for efficient retrieval.

3. **Embeddings for Semantic Search**: The app uses Hugging Face’s `all-MiniLM-L6-v2` model for generating document embeddings. These embeddings enable semantic search, allowing the chatbot to understand and retrieve relevant information from the PDFs.

4. **RAG Pipeline**: The core functionality is based on a Retrieval-Augmented Generation (RAG) architecture, where the chatbot retrieves the most relevant chunks from the PDF content and generates concise, context-aware answers based on those retrieved pieces of information.

5. **Context-Aware Conversations**: The chatbot maintains a history-aware interaction, understanding the context of the conversation by reformulating user questions based on previous messages to ensure continuity in the dialogue.

6. **Seamless User Interface**: The chatbot interface is intuitive, displaying a chat-like experience where users can ask questions and receive answers from the assistant in real-time.

7. **Session Management**: Each user session is uniquely identified, ensuring that chat history and PDF context are preserved during the interaction. This allows for smooth multi-turn conversations without losing track of the previous questions or responses.

### Chatbot Flow

1. **Initialization**: Upon launching the application, users are greeted with a welcome message and a prompt to upload their PDF files via the sidebar.

2. **PDF Processing**: Once the PDFs are uploaded, the documents are split into chunks using the Recursive Character Text Splitter for efficient retrieval.

3. **User Interaction**: Users can type their questions into the chat input box, and the chatbot will analyze the query, retrieve the relevant information from the uploaded PDFs, and generate a concise response.

4. **Contextual Answers**: The chatbot is designed to maintain conversation context, ensuring it understands follow-up questions by reformulating queries based on the chat history.

