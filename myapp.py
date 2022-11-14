import streamlit as st
import pandas as pd
import re
import textProcess as tp
from transformers import BartTokenizerFast, BartForConditionalGeneration
import torch
from PyPDF2 import PdfFileReader,PdfReader
from datetime import datetime
from gsheetsdb import connect
# importing required modules 
import PyPDF2 

import time

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

#Create a connection object.
conn = connect()

@st.cache(ttl=600)
def run_query(query):
    rows = conn.execute(query, headers=1)
    rows = rows.fetchall()
    return rows

@st.experimental_singleton
def get_model(model_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return BartForConditionalGeneration.from_pretrained(model_type).to(device)

@st.experimental_singleton
def bert_smallbert2bert(text):
  checkpoint = "sshleifer/distilbart-cnn-12-6"
  tokenizer = BartTokenizerFast.from_pretrained(checkpoint)
  model = get_model(checkpoint)
  # cut off at BERT max length 512
  inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
  input_ids = inputs.input_ids.to(device)
  attention_mask = inputs.attention_mask.to(device)

  output = model.generate(input_ids, attention_mask=attention_mask)

  return tokenizer.decode(output[0], skip_special_tokens=True)


@st.cache
def get_info(f):
     
    pdf = PdfFileReader(f)
    info = pdf.getDocumentInfo()
    number_of_pages = pdf.getNumPages()
    
    # print(info)
    author = info.author
    subject = info.subject
    title = info.title
    date = info.creation_date
    year=date.strftime('%Y')
    return author,subject,title,year

@st.cache
def extract_abstract(source):
  reader = PdfReader(source)
  page = reader.pages[0]
  text=page.extract_text()
  text=tp.remove_newline(text)
  text=tp.remove_miss(text)
  text=tp.remove_whitespace(text)
  abstract=text.split("Abstract")[1].split("Keywords")[0]
  return abstract

@st.cache
def extract_all_text(source):
   text = []
   pdfReader = PyPDF2.PdfFileReader(source)
   for page in range(1, pdfReader.numPages):
     pageObj = pdfReader.getPage(page)
     text.append(pageObj.extractText())
   return text

@st.cache
def compare(text):
    models = ['convolutional neural networks','convolutional neural network' ,'cnn','cnns','recurrent neural networks','recurrent neural network','rnn','rnns','Deep Neural Networks','dnns','svm','consensus algothrim',"federated learning","svr"]
    used=[]
    for model in models:
        if model in text:
            used.append(model)
    return used

@st.cache
def get_model(source):
   text=extract_all_text(source)
   complete = ''.join(text)
   complete = re.sub('\n', ' ', complete)
   complete = complete.lower()
   complete = re.sub(']', '', complete)
   result = compare(complete)
   return result

@st.cache
def main():
  st.title('SUMBUD')
 
  menu=["Domain-select","File-upload"]
  choice=st.sidebar.selectbox("Pick One",menu)
  
  if choice == "Domain-select":
    st.subheader("Domain-Finding")
    with st.sidebar:
      domain = st.text_input('Domain',placeholder="Enter the Domain")
    
    if domain == "":
      st.success('Pick a Domain , And start to fly 😊', icon="✅")
    else:
      
      with st.spinner('Wait for it...'):
           data['Domain']=data['Domain'].apply(lambda a : tp.lowercasing(a))
           domain_pd=data.loc[data['Domain'] == tp.lowercasing(domain)]
           domain_pd['Summarized_Abstract'] = domain_pd['clean_abstract'].apply(bert_smallbert2bert)
           
           domain_pd=domain_pd.drop(["Abstract","Summary","clean_abstract"],axis=1)
           domain_pd=domain_pd.reset_index(drop = True)
           table=st.table(domain_pd)
           time.sleep(5)

      with st.sidebar:
            csv = convert_df(domain_pd)
            st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='domain_find.csv',
            mime='text/csv',
            )
  
  else:
    st.subheader("File uploaded")
    with st.sidebar:
      file=st.file_uploader("Domain",type=['pdf'],label_visibility="collapsed",accept_multiple_files=True)
    
    
    if file == [] :
      st.success('Add A file , And See the Magic 😊', icon="✅")
    else:
      author_list=[]
      title_list=[]
      year_list=[]
      abstract_list=[]
      model_list=[]
      for i in range(len(file)):
        name=file[i]
        if file is not None:
          author,subject,title,year=get_info(name)
          abstract=extract_abstract(name)
          models=get_model(name)
          if title == "":
            title="Title not Found"
          if author == "":
            author="Author not found"
          if year == "":
            year=0
          
          author_list.append(author)
          title_list.append(title)
          year_list.append(year)
          abstract_list.append(abstract)
          model_list.append(models)
    
      with st.spinner('Wait for it...'):
          file_pd=pd.DataFrame({"Title":title_list,"Contributors":author_list,"Abstract":abstract_list,"Year":year_list,"Models":model_list})
          file_pd['Summarized_Abstract'] = file_pd['Abstract'].apply(bert_smallbert2bert)
          file_pd=file_pd.drop(["Abstract"],axis=1)

          
          
          for i in range (len(file_pd)):
            container = st.expander(file_pd["Title"][i],expanded= True)
            container.write("Authors : " + file_pd["Contributors"][i])
            container.write("Published Year : " + file_pd["Year"][i])
            container.write("Summary : " + file_pd["Summarized_Abstract"][i])
            md=file_pd["Models"][i]
            container.info(f"Possible Models Used : {md} " , icon="ℹ️")

          time.sleep(5)
          
      
      with st.sidebar:
            csv = convert_df(file_pd)
            st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='file_upload.csv',
            mime='text/csv',
            )


sheet_url = st.secrets["public_gsheets_url"]
rows = run_query(f'SELECT * FROM "{sheet_url}"')
# data = pd.read_csv("../sumbud/AIML-RawData.xlsx - Sheet1.csv", sep=",")
data=pd.DataFrame(rows)
data=tp.textp(data) 



if __name__=='__main__':
  main()
    




