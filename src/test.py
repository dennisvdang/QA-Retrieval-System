import csv
import json
import os
import pickle
import re
import time

import pandas as pd
from langchain.chains.query_constructor.base import AttributeInfo, load_query_constructor_chain, StructuredQueryOutputParser, get_query_constructor_prompt
from langchain.docstore.document import Document
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.query_constructor.ir import Comparator, Comparison, Operator, Operation
from langchain.retrievers.self_query.chroma import ChromaTranslator
from tqdm import tqdm
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from langchain.chains.query_constructor.base import load_query_constructor_chain



def csv_to_documents(csv_file_path):
    documents = []
    
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            content = ""
            metadata = {}
            
            for key, value in row.items():
                content += f"{key}: {value}\n"
                metadata[key] = value
            
            document = Document(page_content=content.strip(), metadata=metadata)
            documents.append(document)
    
    return documents


def save_documents(documents, file_path):
    """Save the documents to a file"""
    with open(file_path, 'wb') as file:
        pickle.dump(documents, file)


def load_documents(file_path):
    """Load the documents from a file"""
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    return None


def load_query_constructor_chain(llm, document_contents, attribute_info, enable_limit):
    def construct_prompt(attribute_info, question):
        prompt = f"""
Based on the provided attribute information and the user's question, construct a structured query to retrieve relevant documents.

Attribute Information:
{attribute_info}

User Question: {question}

Structured Query:
"""
        return prompt

    def _parse_result(result):
        try:
            # Attempt to parse the result as JSON first
            query_dict = json.loads(result)
            query = query_dict["query"]
            comparisons = []
            for key, value in query_dict.items():
                if key != "query":
                    comparisons.append(Comparison(comparator=Comparator.EQ, attribute=key, value=value))
        except json.JSONDecodeError:
            # If JSON parsing fails, assume result is a plain string and parse manually
            query = result
            comparisons = []
            # Regex to find patterns like doc.attributesManufacturer = 'ATL'
            matches = re.findall(r"doc\.attributes(\w+)\s*=\s*'([^']+)'", query)
            for match in matches:
                attribute, value = match
                comparisons.append(Comparison(comparator=Comparator.EQ, attribute=attribute, value=value))
        
        # Construct filter from comparisons
        if comparisons:
            _filter = Operation(operator=Operator.AND, arguments=comparisons)
            filter_str = ChromaTranslator().visit_operation(_filter)
        else:
            filter_str = {}

        return {"query": query, "filter": filter_str, "comparisons": comparisons}

    return lambda question_dict: _parse_result(llm.invoke(construct_prompt(question_dict["attribute_info"], question_dict["question"])))

# Load and clean csv data
csv_file_path = 'data/Catalog - Probes.csv'
probes_df = pd.read_csv(csv_file_path)
probes_df.fillna("", inplace=True)
probes_df.drop_duplicates(subset="ProbeID", inplace=True)
probes_df.set_index("ProbeID", inplace=True)

probes_df['Make_Model'] = probes_df['Manufacturer'] + ' ' + probes_df['Model']
probes_df.drop(columns=['Description'], inplace=True)

# Initialize OllamaEmbeddings with the specified model and additional parameters
ollama_emb = OllamaEmbeddings(model="mxbai-embed-large")

# File path for saving/loading embedded documents
embedded_docs_file = 'data/embeddings/embedded_documents.pkl'

# Load the embedded documents if they exist
documents = load_documents(embedded_docs_file)

if documents is None:
    # Create documents
    documents = csv_to_documents(csv_file_path)

    # Save the embedded documents to a file
    save_documents(documents, embedded_docs_file)
    print("Embedded documents saved.")
else:
    print("Loaded embedded documents from file.")

# Path to the directory where the vector store will be persisted
persist_directory = 'data/vectorstore'

# Check if the vector store already exists
if os.path.exists(persist_directory):
    # Load the saved vector store
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=ollama_emb)
    print(f"Loaded vectorstore from {persist_directory}")
else:
    # Initialize Chroma with the OllamaEmbeddings instance directly and the persist_directory
    vectorstore = Chroma(embedding_function=ollama_emb, persist_directory=persist_directory)
    print("Vector store initialized.")

    # Add documents to the vector store
    start_time = time.time()
    print("Adding documents to vector store...")
    document_ids = vectorstore.add_documents(documents)
    print(f"Added documents to Chroma (Completed in {time.time() - start_time:.2f} seconds)")

    # No need to manually persist the vector store, as it is handled automatically
    print(f"Vectorstore saved to {persist_directory}")

metadata_field_info = [
    AttributeInfo(
        name="Manufacturer",
        description="The company that made the ultrasound probe.",
        type="string"
    ),
    AttributeInfo(
        name="Probe_Model",
        description="The name of the ultrasound probe model",
        type="string"
    ),
    AttributeInfo(
        name="Connection_Type",
        description="The type of connector or port used to connect the probe to the ultrasound system and differentiate probes with the same model name and manufacturer.",
        type="string"
    ),
    AttributeInfo(
        name="Compatible_Systems",
        description="Ultrasound systems that work with the specific probe model",
        type="string"
    ),
    AttributeInfo(
        name="Array_Type",
        description="The array type of the probe that determines how the ultrasound beam is shaped, steered, and focused.",
        type="string"
    ),
    AttributeInfo(
        name="Frequency_Range",
        description="The range of frequencies (measured in MHz) at which the probe can operate to produce ultrasound waves",
        type="string"
    ),
    AttributeInfo(
        name="Applications",
        description="The specific medical or diagnostic applications for which the ultrasound probe is designed",
        type="string"
    ),
    AttributeInfo(
        name="Stock",
        description="The number of units of the specific probe model that are currently in stock or available for sale",
        type="string"
    )
]

llm = Ollama(model="llama3")

document_content_description = "Specifications and characteristics of the ultrasound probe/transducer."

query_constructor_chain = load_query_constructor_chain(
    llm=llm,
    document_contents=document_content_description,
    attribute_info=metadata_field_info,
    enable_limit=False
)

start_time = time.time()
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    query_constructor_chain=query_constructor_chain,
    verbose=True,
    use_original_query=True
)
print(f"Initialized SelfQueryRetriever (Completed in {time.time() - start_time:.2f} seconds)")

# Load the QA dataset
qa_df = pd.read_csv('data/qa_dataset.csv')

# Process each question in the DataFrame
for index, row in tqdm(qa_df.iterrows(), total=qa_df.shape[0], desc="Processing QA"):
    question = row['Question']
    question_dict = {"attribute_info": metadata_field_info, "question": question}
    
    # Construct and parse the query
    result = query_constructor_chain(question_dict)
    query = result["query"]
    _filter = result["filter"]
    comparisons = result["comparisons"]
    
    # Print the structured query, filter, comparators, and operators for debugging
    print(f"Structured Query: {query}")
    print(f"Filter: {_filter}")
    print(f"Comparisons: {comparisons}")
    
    question_dict = {"input": {"query": query, "filter": _filter}}
    result = self_query_retriever.invoke(question_dict)
    
    # Assuming result is a list of Document objects
    if result and isinstance(result, list):
        # Extract the answer from the first result
        answer = result[0].page_content if result else ''
        qa_df.loc[index, 'generated_answer'] = answer
        
        # Extract source documents
        source_documents = " | ".join([doc.page_content for doc in result])
        qa_df.loc[index, 'source_documents'] = source_documents
    else:
        # Handle the case where result is empty or not a list
        qa_df.loc[index, 'generated_answer'] = ''
        qa_df.loc[index, 'source_documents'] = ''
    
    # Save the DataFrame to a CSV file after each iteration
    qa_df.to_csv('data/answers_progress.csv', index=False)

# Calculate Shannon entropy



# Save the final DataFrame to a CSV file
#qa_df.to_csv('data/answers.csv', index=False)