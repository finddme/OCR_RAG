from ocr.ocr_processing import ocr_processing
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
from typing import List
from langchain_core.prompts import ChatPromptTemplate
import weaviate, os
import openai
start_page, end_page= 1, 10
weaviate_client = weaviate.Client("http://192.168.0.202:8080") 

class GraphState(TypedDict):
    question : str
    file_name : str
    class_name : str
    retrieve_res: List[dict]
    response: str
    annotations: List[dict]

def pdf_processing(state):
    global start_page
    global end_page
    print("--- PDF processing ---")
    question = state["question"]
    file_name = state["file_name"]
    class_name=ocr_processing(file_name,start_page, end_page)
    return {"question": question, "file_name":file_name, "class_name":class_name}

        
def get_embedding(text, engine="text-embedding-3-large") : 
    Openai_API_KEY = ""
    os.environ["OPENAI_API_KEY"] =  Openai_API_KEY
    openai.api_key =os.getenv("OPENAI_API_KEY")

    # res = openai.Embedding.create(input=text,engine=engine)['data'][0]['embedding']
    from openai import OpenAI
    embedding_client = OpenAI()
    res= embedding_client.embeddings.create(input = text, model=engine).data[0].embedding
    return res
    
def retrieve(state):
    global weaviate_client
    print("--- RETRIEVE from Vector Store DB ---")
    question = state["question"]
    file_name = state["file_name"]
    weaviate_class = state["class_name"]
    property_list = list(map(lambda x: x["name"], weaviate_client.schema.get(weaviate_class)['properties']))
    query_vector = get_embedding(question)
    documents = weaviate_client.query.get(weaviate_class, property_list).with_hybrid(question, vector=query_vector).with_limit(3).do()
    retrieve_res=list(map(lambda x: x["content"], documents["data"]["Get"][weaviate_class.title()]))
    annotations=list(map(lambda x: {"page_num":x["page_num"],"bbox":x["bbox"].split("//"), "box_color":x["box_color"].split("//")}, documents["data"]["Get"][weaviate_class.title()]))
    return {"question": question, "file_name":file_name, "class_name":weaviate_class, "retrieve_res": retrieve_res, "annotations":annotations}

Openai_API_KEY = ""
from openai import OpenAI
llm_client = OpenAI(api_key=Openai_API_KEY)

def get_summary_response(text,documents):
    gpt_prompt=f"""You are a highly knowledgeable and friendly chatbot designed to answer questions based on a provided document about prompt engineering. 
    The document includes information on various aspects of prompt engineering, such as techniques, strategies, use cases, and best practices.
    When a user asks a question, your goal is to provide accurate, concise, and contextually relevant answers based on the document. 
    If the document contains multiple sections that could answer the question, summarize the relevant sections.
    Question: {text}
    
    Here is the document content for reference:
    {documents}

    Guidelines:
    - Refer to the above document and write the answer to the user's question in Korean. All responses must be given in Korean.
    - Always refer to the document content for your answers.
    - Provide clear and concise explanations.
    - If a question is not directly answerable from the document, acknowledge the limitation and suggest consulting the full document for detailed information.

    """
    # response = openai.ChatCompletion.create(
    response = llm_client.chat.completions.create(
        model="gpt-4o",
        # api_key=Openai_API_KEY,
        messages=[{"role": "system", "content": gpt_prompt}],
        max_tokens=1024)
    return response.choices[0].message.content

def generate(state):
    print("--- GENERATE Answer ---")
    question = state["question"]
    file_name = state["file_name"]
    weaviate_class = state["class_name"]
    retrieve_res=state["retrieve_res"]
    annotations=state["annotations"]
    generation= get_summary_response(question,retrieve_res)
    return {"question": question, "file_name":file_name, "class_name":weaviate_class, "retrieve_res": retrieve_res, "annotations":annotations, "response":generation}

def rag_graph():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("pdf_processing", pdf_processing)
    workflow.add_node("retrieve", retrieve) # retrieve
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "pdf_processing")
    workflow.add_edge("pdf_processing", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    app = workflow.compile()

    return app