---
layout: page
title: homework 7
permalink: /homework-7/
description: attention neural networks
nav: true
hw_pdf: assignment-7.pdf
assigned: march 20
due_date: april 3
horizontal: false
---

<hr style="border:2px solid gray">
#### attention and the transformer
-----

We will be summarizing text in this homework assignment.





-----
#### data and starter kit
-----


If you're on a GCP VM, you can download with the bash command `wget`. If you wish to download inside of a notebook, you can type in a shell commnad with a `!`. The full package can be downloaded with

```bash
wget -nc https://course.ccs.neu.edu/cs6120s25/data/samsum/utils.py
wget -nc https://course.ccs.neu.edu/cs6120s25/data/samsum/corpus.tar
tar -xvf corpus.tar
pip install dlai_grader
```

This set of files contains:

* **The SAMsum dataset**: around 16k paired conversations with their human-generated summaries, and can be found [here](https://course.ccs.neu.edu/cs6120s25/data/samsum/). Both the samples and their annotations are created by linguists, reflecting real-life messenger conversations: varying style, formality, slang, emojis, and general language patterns. 

* **Loading Scripts** found in [`utils.py`](https://course.ccs.neu.edu/cs6120s25/data/samsum/utils.py). The dataset has several functions that we will be using to process the data, including splitting training and test data from a folder name and preprocessing that data. We will be calling the majority of these utils functions from the `preprocess_data` function.

* The python code you will need to modify and turn in is located [here]({{ site.baseurl }}/assets/python/assignment6.py). There are several functions that you will be modifying. They will be of the form

  ```python
  def some_function(argument):
    '''Description of arguments and return values
    '''
    ### START CODE HERE ###

    return_values = "This is where you will add or edit the code"

    ### END CODE HERE ###
    return return_values
  ```

  You will need to edit between `START CODE HERE` and `END CODE HERE`.

* Document templates can be either [Overleaf TeX File](https://www.overleaf.com/read/zfwcfsbbgtxj) or [DOCX File](https://docs.google.com/
document/d/1qXipr5Ko2Xpf71GbLzEZXa9khB5w4O2B/edit?usp=sharing&ouid=117230435864186314036&rtpof=true&sd=true). When you've compiled/finishe
d writing, **download the PDF** from Overleaf/Google and upload it to the submission link. 

* You will be turning in Python files, but feel free to develop in notebooks. To set one up, you can use, there are several options:
  * [Locally on Your Laptop](https://jupyter.org/install)
  * [Google Cloud Vertex Work](https://console.cloud.google.com/vertex-ai/workbench) with your Google Cloud credits. 
  * [Google Colabs](https://colab.research.google.com/) with your Google Account

<br>
<br>

#### submission instructions

Submit your work to [Gradescope](http://gradescope.com). You will need to submit the files:

* **assignment6.py** - your solutions to the questions
* **assignment6.h5** - parameters to your model

