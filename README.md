## Design and Implementation of LangChain Expression Language (LCEL) Expressions

### AIM:
To design and implement a LangChain Expression Language (LCEL) expression that utilizes at least two prompt parameters and three key components (prompt, model, and output parser), and to evaluate its functionality by analyzing relevant examples of its application in real-world scenarios.

### PROBLEM STATEMENT:
Design an LCEL pipeline using LangChain with at least two dynamic prompt parameters. Integrate prompt, model, and output parser components to form a complete expression. Evaluate its functionality through real-world query-response scenarios.

### DESIGN STEPS:

#### STEP 1: Setup API and Environment: Load environment variables using dotenv and set openai.api_key from the local environment.
#### STEP 2: Create Prompt and Model: Use LangChain to define a ChatPromptTemplate and initialize ChatOpenAI for text generation.
#### STEP 3: Build a Retrieval System: Store predefined texts in DocArrayInMemorySearch with OpenAIEmbeddings and create a retriever.
#### STEP 4: Define Question-Answering Chain: Use RunnableMap to fetch relevant documents and pass them to a chat model for responses.
#### STEP 5: Invoke the Chain: Run chain.invoke() with a question to retrieve context-based answers using the LangChain pipeline.

### PROGRAM:
### Name : NETHRA.K
### Reg no : 212224230184
### Simple Chain
```

import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) 
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

print("Name : Nethra K\nReg No: 212224230184")

prompt = ChatPromptTemplate.from_template(
    "Explain {concept} in a way that a {audience} can understand."
)
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke({"concept": "Machine Learning", "audience" : "ADVANCE  "})


```

### Complex Chain
```
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch   

vectorstore = DocArrayInMemorySearch.from_texts(
    ["Generative AI is a type of artificial intelligence that creates new content, such as text, images, or code, based on learned patterns.","(LCEL)LangChain Expression Language (LCEL) is a declarative syntax for composing and chaining LangChain components efficiently."],
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

retriever.get_relevant_documents("what is generative ai?")
retriever.get_relevant_documents("what is the full form of LCEL")

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

print("Name : Nethra K\nReg No: 212224230184")
prompt = ChatPromptTemplate.from_template(template)

from langchain.schema.runnable import RunnableMap

chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser

chain.invoke({"question": "what is the full form of LCEL?"})

inputs = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
})

inputs.invoke({"question": "what is the full form of LCEL?"})
```

### OUTPUT:
#### Simple chain
<img width="1311" height="212" alt="image" src="https://github.com/user-attachments/assets/4eb236c6-85e2-429c-8038-a1131ac2dc62" />

#### Complex Chain
<img width="1303" height="190" alt="image" src="https://github.com/user-attachments/assets/45631206-508a-4e25-aa07-71574eef994d" />


### RESULT:
The implemented LCEL expression takes at least two prompt parameters, processes them using a model, and formats the output with a parser, demonstrating its effectiveness through real-world examples.
