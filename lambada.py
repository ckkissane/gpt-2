from gpt2 import GPT2, load_weights
import jsonlines
import transformers
import torch as t

tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")


def preprocess(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("''", '"')
    text = text.replace("``", '"')
    return "\n" + text.strip()


stopwords = {
    "ourselves",
    "hers",
    "between",
    "yourself",
    "but",
    "again",
    "there",
    "about",
    "once",
    "during",
    "out",
    "very",
    "having",
    "with",
    "they",
    "own",
    "an",
    "be",
    "some",
    "for",
    "do",
    "its",
    "yours",
    "such",
    "into",
    "of",
    "most",
    "itself",
    "other",
    "off",
    "is",
    "s",
    "am",
    "or",
    "who",
    "as",
    "from",
    "him",
    "each",
    "the",
    "themselves",
    "until",
    "below",
    "are",
    "we",
    "these",
    "your",
    "his",
    "through",
    "don",
    "nor",
    "me",
    "were",
    "her",
    "more",
    "himself",
    "this",
    "down",
    "should",
    "our",
    "their",
    "while",
    "above",
    "both",
    "up",
    "to",
    "ours",
    "had",
    "she",
    "all",
    "no",
    "when",
    "at",
    "any",
    "before",
    "them",
    "same",
    "and",
    "been",
    "have",
    "in",
    "will",
    "on",
    "does",
    "yourselves",
    "then",
    "that",
    "because",
    "what",
    "over",
    "why",
    "so",
    "can",
    "did",
    "not",
    "now",
    "under",
    "he",
    "you",
    "herself",
    "has",
    "just",
    "where",
    "too",
    "only",
    "myself",
    "which",
    "those",
    "i",
    "after",
    "few",
    "whom",
    "t",
    "being",
    "if",
    "theirs",
    "my",
    "against",
    "a",
    "by",
    "doing",
    "it",
    "how",
    "further",
    "was",
    "here",
    "than",
}

if __name__ == "__main__":
    model = load_weights(GPT2)
    correct, total = 0, 0
    with jsonlines.open("lambada_test.jsonl") as reader:
        for obj in reader:
            text = preprocess(obj["text"])
            tokens = t.tensor(tokenizer.encode(text)[:-1], dtype=t.long).unsqueeze(0)
            final_token = tokenizer.encode(text)[-1]
            gpt_output = model(tokens)
            _, line_encoded_candidates = t.topk(gpt_output.logits, k=20, dim=-1)
            line_encoded_candidates = list(line_encoded_candidates[0])

            predicted = None
            for candidate in line_encoded_candidates:
                if not (tokenizer.decode(candidate).strip() in stopwords):
                    predicted = candidate
                    break

            assert predicted is not None
            total += 1
            if predicted.item() == final_token:
                correct += 1
            if total % 10 == 0:
                print(f"accuracy: {correct / total}")

    print(f"accuracy: {correct / total}")
