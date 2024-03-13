
To get the default used embeddings model :

### Install Git LFS
git lfs install

### Clone the repository
git clone https://huggingface.co/intfloat/multilingual-e5-large



To run everything:

pip install -r requirements.txt

flask --app flask_literature_review.py
streamlit run literature_review_interface.py

Configurations in the config.json file: 

- "embeddings_model": Name of the embeddings model you want to load (in case you already use one)
- "max_chunk_size": Max text size to embedd -  you might want to check your maximum chunk size if you use a different embeddings model than the default,
- "parser_chunk_overlap": How many tokens to overlap when processing the data
- "vector_db_name": name for your chromadb persistent folder
- "top_k": number of chunks to be retrieved and used as context for the llm

### If you choose your own embeddings model, review average_pool() function from flask_litarature_review.py

# Important: after you upload the file and you get a Tost with "File uploaded" -  you have to click on "(X)" on the right of the file in order to finish the upload and continue the QnA