#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import requests
from dotenv import load_dotenv, set_key
from PyPDF2 import PdfFileReader

from bs4 import BeautifulSoup
import streamlit as st
from typing import List
import openai
import urllib.parse

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

if not openai.api_key:
    openai.api_key = input("Enter OPENAI_API_KEY API key")
    set_key(".env", "OPENAI_API_KEY", openai.api_key)

os.environ["OPENAI_API_KEY"] = openai.api_key
input_dir = "input"
output_dir = "output"
text_dir = output_dir + "/text"
done_dir = output_dir + "/done"

for directory in [input_dir, output_dir, text_dir, done_dir]:
    os.makedirs(directory, exist_ok=True)

def get_links_from_url(url):
    pdf_links = []
    try:
        response = requests.get(url)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type")
        if "pdf" in content_type:
            # Direct PDF link
            parsed_url = urllib.parse.urlparse(url)
            filename = os.path.basename(parsed_url.path)
            with open(os.path.join(input_dir, filename), "wb") as f:
                f.write(response.content)
            pdf_links.append(os.path.join(input_dir, filename))
        else:
            # HTML page
            soup = BeautifulSoup(response.content, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                if href.endswith(".pdf"):
                    parsed_url = urllib.parse.urlparse(href)
                    filename = os.path.basename(parsed_url.path)
                    with open(os.path.join(input_dir, filename), "wb") as f:
                        f.write(requests.get(href).content)
                    pdf_links.append(os.path.join(input_dir, filename))
    except Exception as e:
        st.error(f"Error retrieving PDF links from URL: {e}")
    return pdf_links

def get_links_from_file(file: str) -> List[str]:
    with open(file, "r") as f:
        links = [line.strip() for line in f.readlines() if line.strip().endswith(".pdf")]
    return links

def get_links_from_directory(directory: str) -> List[str]:
    pdf_links = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_links.append(os.path.join(directory, filename))
    return pdf_links

def pdf_to_text(pdf_filename, ignore_words=["Yes", "No"]):
    with open(pdf_filename, "rb") as f:
        pdf_reader = PdfFileReader(f)
    
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + " "
    words = [word for word in text.split() if word not in ignore_words]
    return " ".join(words)

st.title("PDF2OpenAI")

pdf_links = []
pdf_link_input = st.text_input("Enter a link to hunt for PDF files").strip()
if pdf_link_input:
    pdf_links.extend(get_links_from_url(pdf_link_input))


pdf_file_upload = st.sidebar.file_uploader("Upload a PDF file:", type="pdf")
if pdf_file_upload is not None:
    with open(os.path.join(input_dir, pdf_file_upload.name), "wb") as f:
        f.write(pdf_file_upload.getbuffer())
    pdf_links.append(os.path.join(input_dir, pdf_file_upload.name))

url_file_upload = st.sidebar.file_uploader("Upload a text file containing URLs to PDFs:", type="txt")
if url_file_upload is not None:
    with open(os.path.join(input_dir, url_file_upload.name), "wb") as f:
        f.write(url_file_upload.getbuffer())
    pdf_links.extend(get_links_from_file(os.path.join(input_dir, url_file_upload.name)))

directory_upload = st.sidebar.file_uploader("Upload a directory containing PDFs:", type=None)
if directory_upload is not None:
    directory_path = os.path.join(input_dir, directory_upload.name)
    os.makedirs(directory_path, exist_ok=True)
    for file in directory_upload:
        with open(os.path.join(directory_path, file.name), "wb") as f:
            f.write(file.getbuffer())
    pdf_links.extend(get_links_from_directory(directory_path))

ignore_words = ["Yes", "No"]
for pdf_link in pdf_links:
    if pdf_link.startswith("http"):
        response = requests.get(pdf_link)
        filename = os.path.basename(pdf_link)
        pdf_path = os.path.join(input_dir, filename)
        with open(pdf_path, "wb") as f:
            f.write(response.content)
    else:
        pdf_path = pdf_link
    with open(pdf_path, "rb") as f:
        pdf_reader = PdfFileReader(f)
        output_text = ""
        for page_num in range(pdf_reader.getNumPages()):
            page_text = pdf_reader.getPage(page_num).extractText()
            for word in page_text.split():
                if word not in ignore_words:
                    output_text += word + " "
            output_text += "\n"
        output_file_path = os.path.join(text_dir, os.path.splitext(os.path.basename(pdf_path))[0] + ".txt")
        with open(output_file_path, "w") as f:
            f.write(output_text)
        st.success(f"Successfully converted {pdf_path} to {output_file_path}")
        output_file_path = os.path.join(text_dir, os.path.splitext(os.path.basename(pdf_path))[0] + ".txt")
model_options = {

    'text-davinci-003': 'text-davinci-003',
    'text-davinci-002': 'text-davinci-002',
    'text-davinci-edit-001': 'text-davinci-edit-001',
    'code-davinci-edit-001': 'code-davinci-edit-001',
    'gpt-3.5-turbo': 'gpt-3.5-turbo',
    'gpt-3.5-turbo-0301': 'gpt-3.5-turbo-0301',
    'gpt-4': 'gpt-4',
    'gpt-4-0314': 'gpt-4-0314'
}

selected_model = st.sidebar.selectbox('Select a model:', list(model_options.keys()))
st.write(f'Selected model: {selected_model}')

headers = {'Authorization': f'Bearer {openai.api_key}'}
responses = []
chunk_size = 2500

with open(output_file_path, "r") as text_file:
    text = text_file.read()
    generated_text_chunks = []
    num_chunks = len(text) // chunk_size + (len(text) % chunk_size > 0)
    st.info(f"{num_chunks} chunks will be submitted to OpenAI") 
    for i in range(0, len(text), chunk_size):

        chunk = text[i:i+chunk_size]
        chunk_length = len(chunk)
        
        prompt = f"Read contents of {chunk}, summarize what you think the risk is with details"
        response = openai.Completion.create(
            engine=model_options[selected_model],
            prompt=prompt,
            max_tokens=2024,
            n=1,
            stop=None,
            temperature=1.0,
        )
        generated_text_chunks.append(response.choices[0].text.strip())
        generated_text = '\n'.join(generated_text_chunks)
        
        break
st.write(generated_text)
           
done_path = os.path.join(done_dir,  "OpenAI_" + os.path.splitext(os.path.basename(pdf_path))[0] + ".txt")
with open(done_path, "w") as done_file:
    done_file.write(generated_text)
