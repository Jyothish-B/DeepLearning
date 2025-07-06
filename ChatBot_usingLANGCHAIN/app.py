import os
from apikey import apikey
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain , SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
os.environ['OPENAI_API_KEY'] = apikey
st.title('UTUBE Summarizer GPT')

prompt = st.text_input("Type input text HERE") 
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='write me a youtube video title about {topic}'
 )

script_template = PromptTemplate(
 input_variables = ['title','wikipedia_research'] ,
 template='write me a youtube video script based on this title TITLE: {title} using this wikipedia reserach info:{wikipedia_research} ')

title_memory = ConversationBufferMemory(input_key= 'topic' , memory_key = 'chat_history')
script_memory = ConversationBufferMemory(input_key= 'title' , memory_key = 'chat_history')

llm = ChatOpenAI(temperature = 0.9,model="gpt-4o-mini" , api_key= apikey )
title_chain = LLMChain(llm =llm , prompt = title_template , output_key = "title" , memory = title_memory)
script_chain = LLMChain(llm = llm , prompt = script_template , output_key = "script",memory = script_memory)



wiki = WikipediaAPIWrapper()
if prompt: 
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title) 
    st.write(script) 

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('Script History'): 
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)
