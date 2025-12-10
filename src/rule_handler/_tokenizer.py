from transformers import GPT2TokenizerFast
import os


class Chunker:
    def __init__(self, max_tokens=800, overlap=100):
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def chunk_text(self, lines):
        chunks = []
        current_chunk = []
        current_token_counts = []
        current_len = 0

        for line in lines:
            tokens = self.tokenizer.encode(line)
            token_count = len(tokens)
            if current_len + token_count <= self.max_tokens:
                current_chunk.append(line)
                current_token_counts.append(token_count)
                current_len += token_count
                continue
            chunks.append(" ".join(current_chunk))
            if self.overlap > 0 and len(current_chunk) > 0:
                current_chunk = current_chunk[-self.overlap :] + [line]
                current_token_counts = current_token_counts[-self.overlap :] + [
                    token_count
                ]
                current_len = sum(current_token_counts)
                continue
            current_chunk = [line]
            current_token_counts = [token_count]
            current_len = token_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def save_chunks(self, chunks, output_path, delimiter="\n\n---\n\n"):
        for i, chunk in enumerate(chunks):
            open(f"{output_path}/chunk_{i}.txt", "a").write(chunk + delimiter)

    def process_file(self, input_path, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        else: 
            for f in os.listdir(output_path):
                os.remove(os.path.join(output_path, f))
        with open(input_path, "r") as f:
            lines = f.readlines()
        chunks = self.chunk_text(lines)
        self.save_chunks(chunks, output_path)
        return chunks


if __name__ == "__main__":
    import sys

    input_file = sys.argv[1]
    output_folder = sys.argv[2]
    chunker = Chunker()
    chunker.process_file(input_file, output_folder)
