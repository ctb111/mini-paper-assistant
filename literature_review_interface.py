import streamlit as st
from io import StringIO
import requests
import warnings
from PyPDF2 import PdfReader
import uuid






warnings.filterwarnings("ignore")

CHAT_URL = "http://127.0.0.1:5000/chat"
ADD_FILE_URL = "http://127.0.0.1:5000/add_from_file"



def extract_text_from_pdf(pdf_file, file_name):

    doc_list = []
    pdf_reader = PdfReader(pdf_file)
    paragraph_idx = 0
    for idx, page in enumerate(pdf_reader.pages):
        paragraph_idx+=1
        #id = str(uuid.uuid4())
        metadata =  {"source": file_name, "page": idx, "start_index": paragraph_idx}
        document = page.extract_text()
        doc_list.append({"document": document, "metadata": metadata}) # build document list

    return doc_list













def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]




# App title
st.set_page_config(page_title="Stanford Demo")



if "docs_processed" not in st.session_state.keys():
    st.session_state.docs_processed = ""


# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


# Replicate Credentials
with st.sidebar:
    st.text('Stanford mini study partner:')
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Sidebar file upload area & logic
    uploaded_file = st.file_uploader("Choose a file", type="pdf")
    if uploaded_file is not None:
        doc_list = extract_text_from_pdf(uploaded_file, uploaded_file.name)

        is_uploaded = requests.post(ADD_FILE_URL, json=dict({"doc_list": doc_list})).json()
        if is_uploaded['response'] == 'ok':
            st.toast('File uploaded')




# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])



# Function for generating LLaMA2 response
def generate_llm_response(prompt_input):
    string_dialogue = ''
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    #the URL of the Flask app
    url = CHAT_URL 

    # Define the question you want to ask
    question = prompt_input

    # Define the payload (question) as a dictionary
    payload = {"question": question}

    # Send the POST request
    response = requests.post(url, json=payload)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Print the response from the chatbot
        print(response.json())
        output = response.json()["response"]
    else:
        # Print an error message if the request was not successful
        print("Error:", response.text)
    return output

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            print(prompt)
            response = generate_llm_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)

    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)