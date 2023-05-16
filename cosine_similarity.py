import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class CosineSimilarityChecker:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.string_list = []
        self.allowed_similarity_threshold: float = 0.7

    def add_string(self, string: str) -> None:
        # Check if the string is too similar to any existing strings
        similarities = [self.cosine_similarity(string, existing_string) for existing_string in self.string_list]
        if any(similarity >= self.allowed_similarity_threshold for similarity in similarities):
            print("String is not allowed due to high similarity.")
        else:
            self.string_list.append(string)
            print("String added successfully.")
        # Output cosine similarity for each input string
        for existing_string, similarity in zip(self.string_list, similarities):
            print(f"Cosine similarity between '{string}' and '{existing_string}': {similarity:.4f}")

    def cosine_similarity(self, string1: str, string2: str) -> float:
        vectors = self.vectorizer.fit_transform([string1, string2]).toarray()
        vector1 = vectors[0]
        vector2 = vectors[1]
        dot_product = np.dot(vector1, vector2)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)
        similarity = dot_product / (norm_vector1 * norm_vector2)
        return similarity

# Example usage
checker = CosineSimilarityChecker()

# Example data
existing_strings = [
    "I love pizza",
    "I enjoy playing soccer",
    "The sun is shining today",
    "Programming is fun"
]
checker.string_list = existing_strings

# Add new strings
new_string1: str = "I like eating pizza"
checker.add_string(new_string1)

new_string2: str = "I enjoy playing basketball"
checker.add_string(new_string2)

new_string3: str = "The weather is nice today"
checker.add_string(new_string3)

new_string4: str = "I love coding"
checker.add_string(new_string4)

new_string5: str = "I love coal"
checker.add_string(new_string5)

new_string6: str = "I love coal"
checker.add_string(new_string6)

new_string7: str = "I love goals"
checker.add_string(new_string7)

new_string8: str = "I love goals"
checker.add_string(new_string8)