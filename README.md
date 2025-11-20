# 📨 Ask Enron – LLM Question Answering over the Enron Email Corpus

**Live demo (Hugging Face Space):** https://huggingface.co/spaces/singhalamaan116/Ask-Enron

Ask questions in natural language and get answers based on real Enron emails.  
Under the hood, the app performs semantic search over the Enron email dataset and 
uses a small language model to generate answers from the retrieved emails.

---

## 🔧 Tech Stack

- **Dataset:** [`corbt/enron-emails`](https://huggingface.co/datasets/corbt/enron-emails)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **LLM:** `google/flan-t5-small`
- **UI:** Gradio (Blocks)
- **Infra:** Hugging Face Spaces (CPU)

---

## 🚀 How it works

1. User enters a question (e.g., *“What trips were people planning in August 2000?”*).
2. The question is embedded with a sentence-transformers model.
3. We compute cosine similarity with precomputed embeddings of ~20k Enron emails.
4. The top-`k` emails are concatenated into a context.
5. A Flan-T5 model reads the context and generates an answer.
6. The app shows:
   - The answer
   - The exact emails used as context

---

## 📁 Project structure

```text
app.py          # Gradio app (Spaces entry point)
requirements.txt
README.md

