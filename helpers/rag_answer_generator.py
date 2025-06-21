from openai import OpenAI

key = "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"


def generate_rag_answer( question, context_chunks, model="gpt-4"):
    client = OpenAI(api_key=key)

    # Step 1: Concatenate top chunks
    context = "\n\n".join([f"{i+1}. {chunk['chunk']}" for i, chunk in enumerate(context_chunks)])

    # Step 2: Construct the prompt
    prompt = f"""You are a helpful assistant. Use the context below to answer the question along with chunk id in the context from where you got the answer.
If the answer is not contained in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:"""

    # Step 3: Generate answer
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return completion.choices[0].message.content.strip()
