# Retrieval Augmented Generation (RAG)

This project implements a Retrieval Augmented Generation (RAG) model for various tasks, combining information retrieval with large language models (LLMs) to enhance response generation quality. It supports retrieval for both PDFs and websites. The models used in this project include **Llama3-8b-8192** for PDF-based tasks and **Mixtral-8x7b-32768** for website-based tasks.

## Prerequisites

To run this project, you'll need to set up a Conda environment, install dependencies, and configure the models. Follow these steps:

### 1. Store API Keys in a `.env` File

You need to create a `.env` file in the root directory of your project to store your API keys securely. This prevents exposing sensitive information like API keys in your codebase.

#### **Steps to Create the `.env` File:**

1. **Get API Keys**:
   - **GroqCloud API Key**: Visit the [GroqCloud API portal](https://groqcloud.com/) to sign up and generate an API key.
   - **Google AI Studio API Key**: Go to the [Google AI Studio console](https://console.cloud.google.com/), create a project, and generate an API key under "APIs & Services" → "Credentials".

2. **Create `.env` File**:
   - In the root of your project directory, create a file named `.env` and add the following content:

     ```bash
     GROQ_API_KEY="your_groqcloud_api_key"
     GOOGLE_API_KEY="your_google_ai_api_key"
     ```

   - Replace `your_groqcloud_api_key` and `your_google_ai_api_key` with the actual keys you obtained.

### 2. Clone the Repository

Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/nafisrabbi/Retrieval_Augmented_Generation.git
```

```bash
cd Retrieval_Augmented_Generation
```

### 3. Create a Conda Environment

Create a Conda environment with Python 3.11 and activate it:

```bash
conda create --name llm_env python=3.11 -y
```

```bash
conda activate llm_env
```

### 4. Install FAISS-GPU

FAISS-GPU is required for fast similarity search and retrieval. Install it using the following command:

```bash
conda install -c conda-forge faiss-gpu -y
```

### 5. Install Dependencies

Install all the required Python dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 6. Running the Application

Once the environment is set up and dependencies are installed, you can run the Streamlit application to interact with the RAG model:

```bash
streamlit run main.py
```

This will launch the application in your default browser. The application is designed to support tasks like retrieving information from PDFs and websites, with the aid of models such as **Llama3-8b-8192** for PDFs and **Mixtral-8x7b-32768** for websites.

## Acknowledgements

- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search.
- [Streamlit](https://streamlit.io/) for building the interactive web application.
- [Llama3](https://ollama.com/) for NLP model integration for PDFs.
- [Mixtral](https://huggingface.co/models) for multilingual and multitask model support for websites.
- [GroqCloud](https://groqcloud.com/) for scalable compute services and API.
- [Google AI Studio](https://console.cloud.google.com/) for AI and machine learning tools.
