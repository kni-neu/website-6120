---
layout: page
title: homework 8
permalink: /homework-8/
description: large language modeling
nav: true
hw_pdf: assignment-7.pdf
assigned: november 14
due_date: november 21
horizontal: false
---

<hr style="border:2px solid gray">
#### building a RAG system with a PEFT model
-----

In this homework, we will be working on tuning and leveraging large language models in a RAG system. Accept the [classroom homework invitation](https://classroom.github.com/a/Kog9MCRN). Review the homework in this [pdf file]({{ site.baseurl }}/assets/pdf/assignment-7.pdf). Remember that reading resources can be found in the [syllabus]({{ site.baseurl }}/syllabus).

-----
#### data and starter kit
-----

This week, we'll be [fine-tuning Llama2](https://towardsdatascience.com/fine-tune-your-own-llama-2-model-in-a-colab-notebook-df9823a04a32) in [this notebook](https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd). You will need [the data](https://course.ccs.neu.edu/cs6220/fall2023/homework-7/). If you are using Colabs (not a requirement), you would need a Google account.

<!--
<center>
<img 
  src="https://images.immediate.co.uk/production/volatile/sites/7/2018/01/TIT011DJ_0-345b632.jpg"
  width="500" height="auto">
</center>
-->

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Load documents
loader = TextLoader("your_document.txt") 
documents = loader.load()

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create embeddings and store in a vector database
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

# Create a RetrievalQA chain
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=db.as_retriever())

# Query the RAG system
query = "What is the main topic of the document?"
result = qa.run(query)
print(result)
```

<br>

Here are the starter kits that you might find useful.

* [Here are some mathematical properties](https://docs.google.com/presentation/d/1zy2veJEjDT-0acPbGsrEC93EP0MOZIx54jL-gA7wPqE) that will help you in derivations.

* Document templates can be either [Overleaf TeX File](https://www.overleaf.com/read/zfwcfsbbgtxj) or [DOCX File](https://docs.google.com/
document/d/1qXipr5Ko2Xpf71GbLzEZXa9khB5w4O2B/edit?usp=sharing&ouid=117230435864186314036&rtpof=true&sd=true). When you've compiled/finishe
d writing, **download the PDF** from Overleaf/Google and upload it to the submission link. 

* To help you get started, you may find that this [Colaboratory Link](https://colab.research.google.com/drive/1dAqxrOEqrvlqhCJ2jwKX4UrDlNNACWC7?usp=sharing) is useful. In order to submit, you'll need to Download the file as a `*.ipynb` file (Under **File** &rarr; **Download** &rarr; **Download .ipynb**.)


#### submission instructions

Commit your `*.py` or `*.ipynb` (either is fine as long as they are commented appropriately), `output.txt`, and `assignment2.pdf` to that 
repository and provide the URL via [Gradescope](https://www.gradescope.com/courses/583114).


<br><br><br>
<hr style="border:2px solid gray">
#### project checkpoint
-----

Each week, there will be a checkpoint for you project so that you are on track to turn in the project at the end of the semester. This week

* Discuss with your peers, form a team that you can work with, and arrive on a topic that you think you could be passionate about. Your team should not be more than four people.

