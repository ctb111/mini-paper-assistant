from flask import Flask, request, jsonify
from transformers import AutoModel, AutoTokenizer
import warnings
import chromadb
import time
import uuid
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch.nn.functional as F
from torch import Tensor
import json
import os
from openai import OpenAI

warnings.filterwarnings("ignore")

with open("config.json") as f:
    project_config = json.load(f)



os.environ["OPENAI_API_KEY"] = project_config['openai_key']
os.environ["TOKENIZERS_PARALLELISM"] = "true"

chroma_client = chromadb.PersistentClient(project_config['vector_db_name'])



CHROMA_RAG_COLLECTION = chroma_client.get_or_create_collection(name="rag_demo_collection")

MAX_CHUNK_SIZE = project_config['max_chunk_size']


################ Embeddings from Hidden State `################################`
def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def compute_usery_query_embedding(tokenizer, model, query):
    batch_dict = tokenizer(query, max_length=MAX_CHUNK_SIZE, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.tolist()[0]


################ Initialize Embeddings Model ################
tokenizer = AutoTokenizer.from_pretrained(project_config['embeddings_model'])
model = AutoModel.from_pretrained(project_config['embeddings_model'])

#############################################################

def chroma_query_top_k(collection, embd, top_k=5):
    res = collection.query(query_embeddings=embd) 
    ids = res['ids'][0][:top_k]
    metadatas = res['metadatas'][0][:top_k]
    documents = res['documents'][0][:top_k]
    return [{"id": id, "document": document, "metadata":metadata} for id, document, metadata in zip(ids, documents, metadatas)]






# Load knowledge vector database

# Define chat chain
template_prompt = """
Use this context in order to answer the question:

{context}

Question: {question}
"""

template_system = """
You a virtual assistant someone in a Stanford class. Answer in the same language the question is and only based on the following context.

Here is an example of how you should respond. If the question has no answer in the context or is not directly answereable using the context, say you do not know the answer for the question
User:

Context [i]: <important information for answering the question> Source: <source of the information>
Question: [question which answer needs to be taken from the context, along with the source]

"""
#model = ChatOpenAI()
client = OpenAI()


# Initialize Flask app
app = Flask(__name__)



@app.route("/", methods=["GET"])
def works():
    return jsonify({"response": "ok"})




@app.route("/add_from_file", methods=["POST"])
def add_from_file():

    MARKDOWN_SEPARATORS = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,  # the maximum number of characters in a chunk: we selected this value arbitrarily
        chunk_overlap=project_config['parser_chunk_overlap'],  # the number of characters to overlap between chunks
        add_start_index=True,  # If `True`, includes chunk's start index in metadata
        strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
        separators=MARKDOWN_SEPARATORS,
    )

    doc_list = request.json.get('doc_list')
    docs_processed = []
    doc_object_list = []

    for doc in doc_list:
        doc_object_list.append(Document(page_content=doc['document'], metadata=doc['metadata']))

    for doc in doc_object_list:
        docs_processed += text_splitter.split_documents([doc])

    embeds= [compute_usery_query_embedding(tokenizer, model, doc.page_content)  for doc in docs_processed]
    
    CHROMA_RAG_COLLECTION.add(
            documents = [doc.page_content for doc in docs_processed],
            embeddings= embeds,
            metadatas= [doc.metadata for doc in docs_processed],
            ids=[str(uuid.uuid4()) for _ in range(len(docs_processed))]
    )
    
    return jsonify({"response": "ok"})



# Define route for handling chat requests
@app.route("/chat", methods=["POST"])
def chat():
    # Get question from request
    question = request.json.get("question")
    print(question)

    # Invoke chat chain with the question
    start = time.time()
    embd = compute_usery_query_embedding(tokenizer, model, question)
    top_k_docs = chroma_query_top_k(CHROMA_RAG_COLLECTION, embd, project_config['top_k'])
    end = time.time()

    print('Elapsed : ', end - start)
    print(top_k_docs[-1])
    context = ''
    for idx, doc in enumerate(top_k_docs):
        context += 'Context {} : {}'.format(idx, doc['document']) + \
            "Source: document: {}, page: {}".format(doc['metadata']['source'], str(doc['metadata']['page'])) + "\n \n"
    print(context)
    print('Message: ', template_prompt.format(context=context, question=question))


    completion = client.chat.completions.create(
       model="gpt-3.5-turbo",
       messages=[
           {"role": "system", "content": template_system},
           {"role": "user", "content": template_prompt.format(context=context, question=question)}
       ]
    )


 
    return jsonify({"response": completion.choices[0].message.content})


if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True)
