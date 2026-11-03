"""Grounded answer generation using the OpenAI chat completions API."""


class OpenAIAnswerGenerator:
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for answer generation")
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def answer(self, question: str, context: str) -> str:
        if not context.strip():
            return "I could not find relevant information in the indexed documents."

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the user's question using only the supplied context. "
                        "If the context does not contain enough evidence, say so clearly. "
                        "Do not invent facts. Cite supporting passages using their bracketed "
                        "labels, such as [1] or [2]."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
        )
        return response.choices[0].message.content or ""
