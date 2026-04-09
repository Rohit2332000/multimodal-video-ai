import time

from dotenv import load_dotenv
import re
import streamlit as st
from torch.nn.functional import embedding

from langchain_huggingface import  HuggingFaceEmbeddings

from youtube_transcript_api.proxies import WebshareProxyConfig

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_chroma import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


# Function to extract video ID from a YouTube URL (Helper Function)
def extract_video_id(url):
    """
    Extracts the YouTube video ID from any valid YouTube URL.
    """
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if match:
        return match.group(1)
    st.error("Invalid YouTube URL. Please enter a valid video link.")
    return None

# Function to get transcript from the video

def get_transcript(video_id, language):
    ytt_api = YouTubeTranscriptApi(
    proxy_config=WebshareProxyConfig(
        proxy_username="oedraiha",
        proxy_password="0cdl5as0m1wy",
    )
)
    try:
        transcript=ytt_api.fetch(video_id,languages=[language])
        full_transcript=" ".join([i.text for i in transcript])
        time.sleep(10)
        return  full_transcript
    except Exception as e:
        st.error(f"Error fetching video {e}")


# Function to translate transcript into english
    #initialize the gemini model
llm=ChatGoogleGenerativeAI(
    model='gemini-2.5-flash-lite',
    temperature=0.2
)

def translate_transcript(transcript):
    try:
        prompt=ChatPromptTemplate.from_template("""
        You are an expert translator with deep cultural and linguistic knowledge.
        I will provide you with a transcript. Your task is to translate it into English with absolute accuracy, preserving:
        - Full meaning and context (no omissions, no additions).
        - Tone and style (formal/informal, emotional/neutral as in original).
        - Nuances, idioms, and cultural expressions (adapt appropriately while keeping intent).
        - Speaker’s voice (same perspective, no rewriting into third-person).
        Do not summarize or simplify. The translation should read naturally in the target language but stay as close as possible to the original intent.

        Transcript:
        {transcript}
        """)
        #Runnable chain

        chain=prompt | llm

        #Run chain

        response=chain.invoke({"transcript":transcript})

        return  response.content

    except Exception as e:
        st.error(f" Error fetching video {e}")


#Function to get important topics

def get_important_topics(transcript):
    try:
        prompt = ChatPromptTemplate.from_template("""
               You are an assistant that extracts the 5 most important topics discussed in a video transcript or summary.

               Rules:
               - Summarize into exactly 5 major points.
               - Each point should represent a key topic or concept, not small details.
               - Keep wording concise and focused on the technical content.
               - Do not phrase them as questions or opinions.
               - Output should be a numbered list.
               - show only points that are discussed in the transcript.
               Here is the transcript:
               {transcript}
               """)

        # Runnable chain
        chain = prompt | llm

        # Run chain
        response = chain.invoke({"transcript": transcript})

        return response.content

    except Exception as e:
        st.error(f"Error fetching video {e}")

#Function to get notes from the video

def generate_notes(transcript):
    try:
        prompt = ChatPromptTemplate.from_template("""
                You are an AI note-taker. Your task is to read the following YouTube video transcript 
                and produce well-structured, concise notes.

                ⚡ Requirements:
                - Present the output as **bulleted points**, grouped into clear sections.
                - Highlight key takeaways, important facts, and examples.
                - Use **short, clear sentences** (no long paragraphs).
                - If the transcript includes multiple themes, organize them under **subheadings**.
                - Do not add information that is not present in the transcript.

                Here is the transcript:
                {transcript}
                """)

        # Runnable chain
        chain = prompt | llm

        # Run chain
        response = chain.invoke({"transcript": transcript})

        return response.content

    except Exception as e:
        st.error(f"Error fetching video {e}")

#Function to create chunks
def create_chunks(transcript):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=2000)
    doc=text_splitter.create_documents([transcript])
    return  doc

#Function to create embedding and store in vector database
def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store=Chroma.from_documents(docs,embeddings)
    return  vector_store

#RAG Function
def rag_answer(question,vectorstore):
    result=vectorstore.similarity_search(question,k=4)
    context_text="\n".join([i.page_content for i in result])

    prompt = ChatPromptTemplate.from_template("""
                    You are a kind, polite, and precise assistant.
                    - Begin with a warm and respectful greeting (avoid repeating greetings every turn).
                    - Understand the user’s intent even with typos or grammatical mistakes.
                    - Answer ONLY using the retrieved context.
                    - If answer not in context, say:
                      "I couldn’t find that information in the database. Could you please rephrase or ask something else?"
                    - Keep answers clear, concise, and friendly.

                    Context:
                    {context}

                    User Question:
                    {question}

                    Answer:
                    """)

    #chain
    chain=prompt | llm

    response=chain.invoke({'context':context_text,"question":question})

    return  response.content

#Function for normal chatting with llm


def normal_chat(user_input, chat_history=None):
    """
    General chat with LLM (no RAG, no tools)
    """

    try:
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful, friendly, and intelligent assistant.

        - Answer clearly and concisely.
        - Be polite and conversational.
        - If needed, explain concepts in simple terms.
        - Maintain context if chat history is provided.

        Chat History:
        {history}

        User:
        {input}

        Assistant:
        """)

        # format history
        history_text = ""
        if chat_history:
            history_text = "\n".join(
                [f"User: {h['user']}\nAssistant: {h['assistant']}" for h in chat_history]
            )

        # chain
        chain = prompt | llm

        response = chain.invoke({
            "input": user_input,
            "history": history_text
        })

        return response.content

    except Exception as e:
        return f"Error: {str(e)}"

def hybrid_chat(question, vectorstore=None, history=None):
    """
    Smart router:
    - Uses RAG if context is relevant
    - Otherwise falls back to normal LLM
    """

    try:
        # If no vector store → normal chat
        if vectorstore is None:
            return normal_chat(question, history)

        # Retrieve context
        docs = vectorstore.similarity_search(question, k=3)
        context = " ".join([doc.page_content for doc in docs])

        # If no useful context → fallback
        if len(context.strip()) < 50:
            return normal_chat(question, history)

        # Ask LLM if context is useful
        check_prompt = f"""
        Check if the context is relevant to answer the question.

        Context:
        {context}

        Question:
        {question}

        Answer only YES or NO.
        """

        decision = llm.invoke(check_prompt).content.lower()

        if "yes" in decision:
            return rag_answer(question, vectorstore)
        else:
            return normal_chat(question, history)

    except Exception as e:
        return f"Error: {str(e)}"