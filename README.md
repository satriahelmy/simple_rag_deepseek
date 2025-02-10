# Simple RAG Implementation

This repository contains an implementation of a Retrieval-Augmented Generation (RAG) system using the following stack:

- **ChromaDB**: Vector database for efficient storage and retrieval of embeddings.
- **Deepseek**: A powerful model for natural language processing and text generation.
- **Gradio**: A simple UI framework to interact with the model.
- **Ollama**: A lightweight framework for managing and running LLMs locally.

## Features
- Retrieval-Augmented Generation for enhanced response accuracy.
- Local embedding storage using ChromaDB.
- Real-time query processing with Deepseek.
- User-friendly web interface powered by Gradio.

## Installation
Ensure you have Python 3.8+ installed, then clone this repository and install dependencies:

```bash
# Clone the repository
git clone [<your-repo-url>](https://github.com/satriahelmy/simple_rag_deepseek.git)
cd simple_rag_deepseek

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

1. Run the RAG pipeline:
```bash
python app.py
```

2. Open the Gradio interface in your browser (usually at `http://localhost:7860`).

## Configuration
You can modify configuration settings in `config.py` (if applicable) to adjust parameters like:
- Database path
- LLM model selection
- Retrieval settings

## Usage
1. Enter a query in the Gradio interface.
2. The system will retrieve relevant context from ChromaDB.
3. Deepseek will generate a response based on retrieved context.
4. The response is displayed in the UI.

## Folder Structure
```
📂 project-root
│-- app.py               # Main application file
│-- requirements.txt     # Required dependencies

## Future Enhancements
- Integration with additional LLMs.
- Optimization of retrieval strategies.
- Fine-tuning Deepseek for domain-specific tasks.

## Acknowledgements
- [ChromaDB](https://github.com/chroma-core/chroma)
- [Deepseek](https://deepseek.com/)
- [Gradio](https://www.gradio.app/)
- [Ollama](https://ollama.ai/)
