---
layout: page
title: homework 6
permalink: /homework-6/
description: designing recurrent networks 
nav: true
hw_pdf: assignment-6.pdf
assigned: March 16, 2026
due_date: March 23, 2026
horizontal: false
---

<hr style="border:2px solid gray">
#### evaluating classifier performance
-----

Named entity recognition is the process of identifying and classifying entities in unstructured text. Review the homework in this [pdf file]({{ site.baseurl }}/assets/pdf/assignment-6.pdf). We will be borrowing the codebase from [DeepLearning.AI](http://deeplearning.ai). Remember that reading resources can be found in the [syllabus]({{ site.baseurl }}/syllabus).

<center>
<img 
  src="{{ site.baseurl }}/assets/img/ner-hw6.png"
  width="750" height="auto">
</center>
<br>

-----
#### data and starter kit
-----

* The code template for homework 6 is located [here]({{ site.baseurl }}//assets/python/assignment6.py). Please fill out where you see `# <YOUR-CODE-HERE>`. There will be `None` values for those lines that require attention.

* The dataset has several different sizes for prototyping, evaluation, and training, and can be found [here](https://course.ccs.neu.edu/cs6120s26/data/named-entities). If you're on a GCP VM, you can download with the bash command `wget`. If you wish to download inside of a notebook, you can type in a shell commnad with a `!`.

* You will need the data and loading scripts, which are found in [this zipfile](https://course.ccs.neu.edu/cs6120s26/data/named-entities/ner-data.zip). The dataset has several different sizes for prototyping, evaluation, and training. You can download it with the `wget` command. Inside of this zip file, there is a script called `load_data`. To make the Python code work, you will put load_data.py where you can import it. Line 12 in `assignment6.py` appears as follows:

  ```python
  from load_data import load_data
  ```

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

* You are free to *develop* in any environment (including [virtual machines](https://console.cloud.google.com/compute/instances) and [notebooks](https://console.cloud.google.com/vertex-ai/workbench)), but your submission must be a `*.py` file.

<br>
<br>

#### submission instructions

Submit your work to [Gradescope](http://gradescope.com). You will need to submit the files:

* **assignment6.py** - your solutions to the questions
* **assignment6.h5** - parameters to your model

