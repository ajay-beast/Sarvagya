
import json
import re
from string import punctuation
from nltk.corpus import stopwords

def updated_sessions_list(json_data_list):
    # Load NLTK stopwords
    stop_words = set(stopwords.words("english"))

    # Function to preprocess text
    def preprocess_text(text):
        # Remove punctuation
        text = "".join([char for char in text if char not in punctuation])
        # Remove stop words and extra spaces
        text = " ".join([word for word in text.split() if word.lower() not in stop_words])
        # Remove extra newlines
        text = re.sub(r'\n+', ' ', text)
        # Convert to alphanumeric lowercase
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
        return text.strip()

    # # Load JSON data
    # with open(filename, 'r') as file:
    #     json_data_list = json.load(file)

    queries = []
    responses = []

    # Iterate through each JSON data
    for json_data in json_data_list:
        # Extract session ID and mappings
        session_id = json_data.get("id")
        mappings = json_data.get("mapping", {})

        session_queries = []
        session_responses = []

        # Iterate through mappings to extract user-assistant interactions
        for key, value in mappings.items():
            if value.get("message") and value["message"]["author"]["role"] == "user":
                user_content = value["message"]["content"]["parts"]
                children_list = value.get("children", [])
                next_author_key = children_list[0] if children_list else None
                next_author_content = mappings.get(next_author_key, {}).get("message", {}).get("content", {}).get("parts", [])
                
                # Preprocess query and response
                preprocessed_query = preprocess_text(user_content[0])
                preprocessed_response = preprocess_text(" ".join(next_author_content))
                
                # Append preprocessed query and response to session lists
                session_queries.append(preprocessed_query)
                session_responses.append(preprocessed_response)

        # Append session lists to main lists
        queries.append(session_queries)
        responses.append(session_responses)

    # Return the processed session data
    return queries, responses

# Example usage:
# filename = 'sample.json'
# queries, responses = updated_sessions_list(filename)
# print(queries)
# print(responses)
# for session_id, session_queries, session_responses in zip(range(1, len(queries) + 1), queries, responses):
#     print("Session ID:", session_id)
#     print("Queries:", session_queries)
#     print("Responses:", session_responses)
#     print()