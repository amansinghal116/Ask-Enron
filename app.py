import numpy as np
import gradio as gr
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -----------------------
# 1. Load dataset (subset)
# -----------------------
# Full dataset ~500K emails; we sample a subset for demo performance.

print("Loading Enron dataset from Hugging Face...")
enron = load_dataset("corbt/enron-emails", split="train[:20000]")  # first 20k emails

# Build a text field: subject + body
texts = []
for subject, body in zip(enron["subject"], enron["body"]):
    subject = subject or ""
    body = body or ""
    txt = f"Subject: {subject}\n\n{body}"
    texts.append(txt)

print(f"Loaded {len(texts)} emails.")

# -----------------------
# 2. Embedding model
# -----------------------
print("Loading embedding model...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Computing embeddings (this may take a bit)...")
embeddings = embedder.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True,
    normalize_embeddings=True,  # so dot product == cosine similarity
)
print("Embeddings ready.")

# -----------------------
# 3. LLM for answering
# -----------------------
print("Loading QA model (Flan-T5-small)...")
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
)


def retrieve_emails(query, top_k=5):
    """
    Retrieve top_k most relevant emails for a given query using cosine similarity.
    """
    query_emb = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    # cosine similarity for normalized vectors is just dot product
    scores = embeddings @ query_emb
    top_idx = np.argsort(-scores)[:top_k]

    results = []
    for i in top_idx:
        row = enron[i]
        score = float(scores[i])
        results.append(
            {
                "subject": row.get("subject", ""),
                "from_": row.get("from", ""),
                "to": row.get("to", []),
                "date": str(row.get("date", "")),
                "body": row.get("body", ""),
                "score": score,
            }
        )
    return results


def build_context_snippet(results, max_chars=3500):
    """
    Build a context string from retrieved emails, truncated to max_chars.
    """
    pieces = []
    for r in results:
        header = f"From: {r['from_']}\nTo: {', '.join(r['to'] or [])}\nDate: {r['date']}\nSubject: {r['subject']}\n"
        body = r["body"] or ""
        snippet = header + "\n" + body.strip()
        pieces.append(snippet)

    context = "\n\n--- EMAIL ---\n\n".join(pieces)
    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n[Context truncated]"
    return context


def answer_question(question, top_k):
    if not question or question.strip() == "":
        return "Please type a question about the Enron emails.", ""

    # 1. Retrieve relevant emails
    results = retrieve_emails(question, top_k=top_k)

    # 2. Build context for the LLM
    context = build_context_snippet(results, max_chars=3500)

    prompt = (
        "You are an assistant answering questions about the Enron email corpus. "
        "Use ONLY the information from the context emails below. If you are unsure, say so.\n\n"
        f"Context emails:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer in a clear paragraph:\n"
    )

    # 3. Generate answer
    out = llm(
        prompt,
        max_new_tokens=256,
        num_beams=4,
        temperature=0.3,
    )[0]["generated_text"].strip()

    # 4. Format retrieved emails as markdown to show below
    md_context = ""
    for i, r in enumerate(results, start=1):
        md_context += f"### Email {i} (score={r['score']:.3f})\n"
        md_context += f"**From:** {r['from_']}  \n"
        md_context += f"**To:** {', '.join(r['to'] or [])}  \n"
        md_context += f"**Date:** {r['date']}  \n"
        md_context += f"**Subject:** {r['subject']}\n\n"
        # Show a shortened body
        body = (r['body'] or "").strip()
        if len(body) > 800:
            body = body[:800] + "... [truncated]"
        md_context += f"```text\n{body}\n```\n\n"

    return out, md_context


# -----------------------
# 4. Gradio UI
# -----------------------
with gr.Blocks(title="Ask Enron – Email QA") as demo:
    gr.Markdown(
        """
        # 📨 Ask Enron – Question Answering over the Enron Email Corpus

        Ask natural-language questions about the Enron email dataset.  
        The system retrieves relevant emails and uses a small language model (Flan-T5) 
        to generate an answer based on those emails.

        ⚠️ **Note:** Answers are based on a subset of the public Enron email corpus and may be incomplete or approximate.
        """
    )

    with gr.Row():
        with gr.Column():
            question_in = gr.Textbox(
                label="Your question",
                lines=3,
                placeholder="e.g. What projects was Phillip Allen working on in October 2000?",
            )
            top_k_in = gr.Slider(
                minimum=3,
                maximum=15,
                value=7,
                step=1,
                label="Number of emails to search",
            )
            ask_btn = gr.Button("Ask Enron")

        with gr.Column():
            answer_out = gr.Markdown(label="Answer")

    gr.Markdown("## Retrieved emails used as context")
    context_out = gr.Markdown()

    ask_btn.click(
        fn=answer_question,
        inputs=[question_in, top_k_in],
        outputs=[answer_out, context_out],
    )

if __name__ == "__main__":
    demo.launch()
