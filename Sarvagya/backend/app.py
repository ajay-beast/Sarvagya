from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS module
from openai import OpenAI
import json
from Preprocess import updated_sessions_list
from Model import createUserEmbedding
from Response_Prediction import getMostPersonalizedResponse
from RAG import rag_model, rag_preprocess, generate_output_file
import os
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

user_embeddings_cache = {}
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "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"

# @app.route('/api/process-query-without-json', methods=['POST'])
# def process_query_without_json():
#     query = data.get
#     try:
#         response = get_chatgpt_responses(query)


@app.route('/api/process-query', methods=['POST'])
def process_query():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Extract history and query from the JSON data
        history = data.get('history')
        query = data.get('query')
        response=''
        if history == '':
            response = get_chatgpt_responses_without_json(query)
        else:
            queries, responses = updated_sessions_list(history)
            queries_key = tuple(map(tuple, queries))
            if queries_key in user_embeddings_cache:
                user_profile = user_embeddings_cache[queries_key]
            else:   
                user_profile = createUserEmbedding(queries,responses)
                user_embeddings_cache[queries_key] = user_profile
                # print(user_profile)

            # Process the data (replace this with your actual processing logic)
            print(f'Received Query: {query}')
            # print(f'Received history: {history}')
            item_list = rag_preprocess(history)
            generate_output_file(item_list)
            responses_list = []
            for i in range(1):
                response = get_chatgpt_responses(query)
                responses_list.append(response)

            response = getMostPersonalizedResponse(query,responses_list,user_embeddings_cache[queries_key])

        result = {'status': 'success', 'res': response}

        # Return the result as JSON
        return jsonify(result)

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def get_chatgpt_responses(prompt, num_responses=1):
    # api_key = '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'
    # client = OpenAI(api_key=api_key)

    # Create a chat completion request
    # chat_completion = client.chat.completions.create(
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": prompt,
    #         }
    #     ],
    #     model="gpt-3.5-turbo",
    # )

    # print(chat_completion)
    # Get the response content
    rag_chain = rag_model()
    response_content = rag_chain.invoke(prompt)
    print(response_content)
    return response_content

def get_chatgpt_responses_without_json(prompt):
    api_key = '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'
    client = OpenAI(api_key=api_key)

    # Create a chat completion request
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )

    # print(chat_completion)
    # Get the response content
    response_content = chat_completion.choices[0].message.content
    return response_content

if __name__ == '__main__':
    app.run(debug=True)
