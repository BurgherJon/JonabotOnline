import warnings
warnings.filterwarnings('ignore')

from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.schema.document import Document
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Bedrock
import gradio as gr
import json
import os
import sys
import boto3
import numpy as np

module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock

boto3_bedrock = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)

#This creates and answers the questions.
def greet(question):

    #Generate the RAG Context
    query_embedding = vectorstore_faiss_jonabot.embedding_function.embed_query(question)
    np.array(query_embedding)
    relevant_documents = vectorstore_faiss_jonabot.similarity_search_by_vector(query_embedding)
    context = ""
    #print(f'{len(relevant_documents)} documents are fetched which are relevant to the query.')
    #print('----')
    for i, rel_doc in enumerate(relevant_documents):
        new_context = " " + rel_doc.page_content + " "
        #print("Doc:  " + new_context)
        #print('---')
        context = context + new_context

    #Assemble the prompt including providing the prompt engineering.
    prompt_data = "You are JonaBot, the AI version of Jonathan Cavell.  JonaBot leads the Apps, Data, and AI Practice of Kyndryl Consult.  JonaBot is a middle-aged IT consultant named Jonathan.  JonaBot uses analogies in most of answers.  JonaBot is an expert in AWS and learning a lot more about Big Data and Artificial Intelligence.  JonaBot is a firm believer in bringing agile and iterative methods to all parts of business.  JonaBot really believes that one way to do this is through making all centralized IT into platforms.  You can create a platform instead of having a set of service catalogs.  These platforms can be fit-for-purpose for the customers of the platforms (usually the application developmers) and created as products by the centralized IT teams.  Examples of these fit-for-purpose platforms that are developed as products are Cloud Platforms, Individual Development Platforms (IDPs), and even ERP or CRM Platforms.  JonaBot really enjoys sports (especially the Pittsburgh Pirates, Pittsburgh Steelers, and Penn State Nittany Lions Football).  JonaBot loves to make analogies, especially sports ones.  JonaBot also LOVES to make Dad Jokes, any kind of funny pun.  JonaBot's comments need to include a lot of puns and analogies.  While JonaBot remains professional he does include a lot of personal information and antecdotes about his life in conversation."
    prompt_data = prompt_data + "/n  <context>" + context + "</context>"
    prompt_data = prompt_data + "/n  Question: " + question
    prompt_data = prompt_data + "/n  Answer: "

    parameters = {
    "maxTokenCount":512,
    "stopSequences":[],
    "temperature":1,
    "topP":0.9
    }

    #Make the request to the LLM.
    body = json.dumps({"inputText": prompt_data, "textGenerationConfig": parameters})
    modelId = "amazon.titan-tg1-large"
    accept = "application/json"
    contentType = "application/json"
    
    #Deal with the the response to the LLM
    response = boto3_bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get("body").read())
    answer = response_body.get("results")[0].get("outputText")
    return answer.strip()


#The examples that I want to provide
examples = []
examples.append("What Scotch would you recommend?")
examples.append("Where do you work?")
examples.append("How should centralized IT transform in an agile world?")


#Build the blocks for the demo.  Note that since I'm planning to embed this on a larger page I'm opting not to use a title or any markdown at the beginning.  I also am formatting it to use the font my website does.
theme = gr.themes.Base(font=[gr.themes.GoogleFont('Montserrat'), 'ui-sans-serif', 'system-ui', 'sans-serif'])
with gr.Blocks(theme=theme) as demo:
    inp = gr.Textbox(placeholder="Type your question for Jonabot here.", label="Human:")
    out = gr.Textbox(label="Jonabot:")
    ask_button = gr.Button("Ask Jonabot!")
    ask_button.click(greet, inp, out)
    gr.Examples(examples, inputs=[inp])


#Setup the LLM for lanchain
model_id = "amazon.titan-tg1-large"
titan_llm = Bedrock(
    client=boto3_bedrock,
    model_id=model_id
)
titan_llm.model_kwargs = {'temperature': 0.9, "maxTokenCount": 700}

#Import the biographical information from s3 and format it for the vector database
s3 = boto3.client('s3')
response = s3.get_object(Bucket='jonobot', Key='Training_Data/Biographical_Info.txt')
strs = response['Body'].read().decode('utf-8').splitlines()
docs = []
for str in strs:
    docs.append(Document(page_content=str))  

#Setup the local Vector Store
br_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=boto3_bedrock)
vectorstore_faiss_jonabot = FAISS.from_documents(
    documents=docs,
    embedding = br_embeddings, 
)
print(f"vectorstore_faiss_jonabot:created={vectorstore_faiss_jonabot}::")
wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss_jonabot)

#Launch the App
if __name__ == "__main__":
    demo.launch()