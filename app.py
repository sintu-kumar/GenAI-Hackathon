# -*- coding: utf-8 -*-
"""
Created on Sat May 27 13:26:06 2023

@author: sintu
"""
import random
import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import openai
#from playsound import playsound
#from gtts import gTTS
from PyPDF2 import PdfReader
from utils import text_to_docs
from langchain import PromptTemplate, LLMChain
#import os
#from io import StringIO
#from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.chains.summarize import load_summarize_chain
#import os
#import pyaudio
#import wave
#import langchain
#from langchain.document_loaders import UnstructuredPDFLoader
#from io import BytesIO
# import streamlit.components.v1 as components
#from st_custom_components import st_audiorec, text_to_docs


#import sounddevice as sd
#from scipy.io.wavfile import write
from usellm import Message, Options, UseLLM

llm = OpenAI(model_name='text-davinci-003', temperature=0.2, max_tokens=512, openai_api_key='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')

def record_audio(filename):
    duration=5
    sample_rate = 44100
    # Record audio
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=6)
    st.write('Recording started. Speak into the microphone...')
    # Wait for recording to complete
    sd.wait()
    # Save the recorded audio to a WAV file
    write(filename, sample_rate, audio_data)

    st.write(f'Recording saved as {filename}')
    
def transcribe_audio(filename):
    with open(filename, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", filename)
        return transcript["text"]


st.title("Smart QnA")

model_name = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_data
def usellm(prompt):

    service = UseLLM(service_url="https://usellm.org/api/llm")
    messages = [
      Message(role="system", content="You are a financial and share market analyst"),
      Message(role="user", content=f"{prompt}"),
      ]
    options = Options(messages=messages)
    response = service.chat(options)
    return response.content


@st.cache_resource
def embed(model_name):
    hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)
    #embedding = OpenAIEmbeddings(openai_api_key='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')
    return hf_embeddings

# @st.cache_resource
# def chain(_llm):
#     chain = load_summarize_chain(llm=_llm, chain_type="map_reduce")
# chain = load_summarize_chain(llm=llm, chain_type="map_reduce")

hf_embeddings = embed(model_name) #OpenAIEmbeddings(openai_api_key='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')

# File Upload
file = st.file_uploader("Upload a file")

st.write(file)
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1500,
    chunk_overlap  = 100,
    length_function = len,
    separators=["\n\n", "\n", " ", ""]
)
#text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=100, chunk_overlap=0)
#texts = ''
@st.cache_data
def embedding_store(file):
    # save file
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    #st.write(text)
    texts =  text_splitter.split_text(text)
    docs = text_to_docs(texts)
    #st.write(texts)
    docsearch = FAISS.from_documents(docs, hf_embeddings)
    return docs, docsearch

# Submit Button
if st.button("Submit"):
    if file is not None:
        # File handling logic
        st.write("File Uploaded...")
        _, docsearch = embedding_store(file)
        queries ="Revenue of the company and business risks?\
        company forcasts and performance in different category or segments if any."
        contexts = docsearch.similarity_search(queries, k=1)
        prompts = f"Give concise answer to the below questions as truthfully as possible as per given context only,\n\n\
              1. Revenue of the company and business risks? \n\
              2. Create sections explaining some chosen subtopics related to the uploaded file.\n\
              3. What are some of the business risk?\n\
              4. Give me product-wise or service-wise performance\
              Context: {contexts}\n\
              Response (in readable bullet points): "
              

        response = usellm(prompts)
        # Print the assistant's response
        st.subheader("Few informative points in the uploaded document are as follows:")
        st.write(response)

#st.write("Uploaded File Contents:")
if file is not None:
    docs, docsearch = embedding_store(file)
#docs = text_to_docs(texts)
#st.write(docs)
#summary = chain.run(docs)
#st.write(summary)
#st.header("Speech and Text Input")
# Speech Input
# st.subheader("Speech Input")
# wav_audio_data = st_audiorec()

# if wav_audio_data is not None: 
#     st.audio(wav_audio_data, format='audio/wav')
#     audio_text = transcribe_audio(wav_audio_data)
#     st.write(audio_text)
    
# if st.button("Voice Input"):
#     rand = random.randint(1, 10000)*random.randint(10001,20000)
#     audio_filename = f"recorded_audio{rand}.wav"
   
#     record_audio(filename=audio_filename)
#     aud = transcribe_audio(audio_filename)
#     st.write(aud)

# Text Input
st.subheader("Ask Questions")
query = st.text_input('your queries will go here...')

def LLM_Response():
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run({"query":query, "context":context})
    return response


        
if query:
    # Text input handling logic
    #st.write("Text Input:")
    #st.write(text_input)
    context = docsearch.similarity_search(query, k=5)
    prompt = f"Act as a financial analyst and give concise answer to below Question as truthfully as possible, with given Context\n\n\
              Question: {query}\n\
              Context: {context}\n\
              Response: "

    #prompt = PromptTemplate(template=prompt, input_variables=["query", "context"])
    response = usellm(prompt) #LLM_Response()
    st.write(response)
    #language = 'en'
    # Create a gTTS object
    #tts = gTTS(text=response, lang=language)
    
    # Save the audio file
    #rand = random.randint(1, 10000)*random.randint(10001,20000)
    #audio_file = f'output{rand}.mp3'
    #tts.save(audio_file)
    #playsound(audio_file)
