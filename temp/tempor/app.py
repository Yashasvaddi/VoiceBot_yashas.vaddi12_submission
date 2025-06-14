from response_generator import get_response

if __name__ == "__main__":
    while True:
        # q = input("Ask a question (or 'exit'): ")
        q="Mujhe kitne paise daalne padenge?"
        if q.lower() == 'exit':
            break
        answer, source, score = get_response(q)
        print(f"\nAnswer ({source}, confidence={score:.2f}%):\n{answer}\n")
