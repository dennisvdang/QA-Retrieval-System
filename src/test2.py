import os

import pandas as pd
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.retrievers import SelfQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain.docstore.document import Document
from langchain.chains.query_constructor.base import load_query_constructor_runnable, get_query_constructor_prompt
from tqdm import tqdm
from langchain.prompts import BasePromptTemplate
from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery
)
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel



# Load and preprocess data
probes_df = pd.read_csv('data/Catalog - Probes.csv')
probes_df.fillna("", inplace=True)
probes_df.drop(columns=['Description'], inplace=True)
probes_df['Compatible_Systems'] = probes_df['Compatible_Systems'].apply(lambda x: [system.strip() for system in x.split(',')])

# Define attributes
attribute_info = [
    {"name": "Manufacturer", "type": "string", "description": "The company that made the ultrasound probe."},
    {"name": "Probe_Model", "type": "string", "description": "The name of the ultrasound probe model."},
    {"name": "Connection_Type", "type": "string", "description": "The type of connector or port used to connect the probe to the ultrasound system."},
    {"name": "Compatible_Systems", "type": "string", "description": "Ultrasound systems that work with the specific probe model."},
    {"name": "Array_Type", "type": "string", "description": "The array type of the probe."},
    {"name": "Frequency_Range", "type": "string", "description": "The range of frequencies at which the probe can operate."},
    {"name": "Applications", "type": "string", "description": "The specific medical applications for which the ultrasound probe is designed."},
    {"name": "Stock", "type": "string", "description": "The number of units of the specific probe model in stock."}
]

# Initialize Ollama
llm = Ollama(model="llama3")

# Define your document content description and metadata field info
document_content_description = "Specifications and characteristics of the ultrasound probe/transducer."

# Define examples
examples = [
    (
        "Who makes the C3 probe?",
        {
            "query": "Manufacturer of C3 probe",
            "filter": "eq(\"Probe_Model\", \"C3\")"
        }
    ),
    (
        "Does the ATL C3 work with the HDI 5000?",
        {
            "query": "ATL C3 compatibility with HDI 5000 ultrasound system",
            "filter": "and(eq(\"Manufacturer\", \"ATL\"), eq(\"Probe_Model\", \"C3\"), eq(\"Compatible_Systems\", \"HDI 5000\"))"
        }
    ),
    (
        "Is C3 made by ATL or G.E.?",
        {
            "query": "Manufacturer of C3 probe",
            "filter": "and(eq(\"Probe_Model\", \"C3\"), or(eq(\"Manufacturer\", \"ATL\"), eq(\"Manufacturer\", \"G.E.\")))"
        }
    )
]

# Define allowed comparators
allowed_comparators = [
    Comparator.EQ,  # Equal
    Comparator.NE,  # Not equal
    Comparator.GT,  # Greater than
    Comparator.GTE, # Greater than or equal to
    Comparator.LT,  # Less than
    Comparator.LTE  # Less than or equal to
]

# Define allowed operators
allowed_operators = [
    Operator.AND,  # Logical AND
    Operator.OR,   # Logical OR
    Operator.NOT,  # Logical NOT
]

class SimpleSchemaPrompt(BasePromptTemplate):
    def generate_prompt(self, allowed_comparators, allowed_operators):
        comparators = ', '.join([comp.value for comp in allowed_comparators])
        operators = ', '.join([op.value for op in allowed_operators])
        return f"Allowed comparators: {comparators}\nAllowed operators: {operators}"

# Create an instance of the schema prompt
schema_prompt = SimpleSchemaPrompt()

prompt = get_query_constructor_prompt(
    document_contents=document_content_description,
    attribute_info=attribute_info,
    examples=None,  # You can define examples if needed
    allowed_comparators=allowed_comparators,
    allowed_operators=allowed_operators,
    enable_limit=False,
    schema_prompt=schema_prompt
)
print(prompt.format(query="{query}"))

chain = load_query_constructor_runnable(
    llm=llm,
    document_contents=document_content_description,
    attribute_info=attribute_info,
    examples=examples,
    fix_invalid=True
)

# Initialize OllamaEmbeddings with the specified model and additional parameters
ollama_emb = OllamaEmbeddings(model="mxbai-embed-large")

# Check if the vector store already exists
vectorstore_path = 'data/vectorstore'
if not os.path.exists(vectorstore_path):
    os.makedirs(vectorstore_path)  # Ensure the directory exists
    # Create documents
    documents = [Document(page_content=str(row.to_dict()), metadata=row.to_dict()) for _, row in probes_df.iterrows()]
    # Initialize vector store
    vectorstore = Chroma(embedding_function=ollama_emb, persist_directory=vectorstore_path)
    print("Adding documents to vector store...")
    document_ids = vectorstore.add_documents(documents)
    print(f"Vectorstore saved to {vectorstore_path}")
else:
    # Load the existing vector store
    vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=ollama_emb)
    print(f"Loaded vectorstore from {vectorstore_path}")

# Initialize the retriever
retriever = SelfQueryRetriever(
    query_constructor=chain,
    vectorstore=vectorstore,
    structured_query_translator=ChromaTranslator(),
    verbose=True
)

# Load the questions dataset
questions_df = pd.read_csv('data/qa_dataset.csv')

# Initialize a new column for storing the results
questions_df['Query_Result'] = None

# Loop through each question and retrieve answers
for index, row in tqdm(questions_df.iterrows(), total=questions_df.shape[0], desc="Processing Questions"):
    result = retriever.invoke(
        {
            "query": row['Question']
        }
    )
    # Store the result in the new column
    questions_df.at[index, 'Query_Result'] = result
    # Save progress to CSV after each query is processed
    questions_df.to_csv('data/structured_queries_progress.csv', index=False)
# Export the updated DataFrame to a CSV file
questions_df.to_csv('data/structured_queries.csv', index=False)