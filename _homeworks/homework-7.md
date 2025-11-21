---
layout: page
title: homework 7
permalink: /homework-7/
description: attention neural networks
nav: true
hw_pdf: assignment-7.pdf
assigned: november 17
due_date: december 1
horizontal: false
---

<hr style="border:2px solid gray">
#### attention and the transformer
-----

Summarization is an important task in natural language processing and could be useful for a consumer enterprise. For example, bots can be used to scrape articles, summarize them, and then you can use sentiment analysis to identify the sentiment about certain stocks. By completing this assignment you will learn to:

* Use built-in functions to preprocess your data
* Implement DotProductAttention
* Implement Causal Attention
* Understand how attention works
* Build the transformer model
* Evaluate your model
* Summarize an article

This model is slightly different than the ones you have already implemented. This is heavily based on attention and does not rely on sequences, which allows for parallel computing.

<center>
<img 
  src="https://deadline.com/wp-content/uploads/2024/09/Transformers-One.jpg"
  width="750" height="auto">
</center>
<br>

-----
#### data and starter kits
-----

The python code you will need to modify and turn in is located [here]({{ site.baseurl }}/assets/python/assignment7.py). There are several functions that you will be modifying. They will be of the form

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

In addition, you will need the data. There are a variety of ways you can download them onto your workstations. On [GCP VMs](https://console.cloud.google.com/compute/instances), you can download with the bash command `wget`. If you wish to download inside of a notebook, you can type in a shell command by prepending each of the following commands with `!`. The full package can be downloaded with

  ```bash
  wget -nc https://course.ccs.neu.edu/cs6120f25/assets/python/assignment7.py
  wget -nc https://course.ccs.neu.edu/cs6120f25/data/samsum/utils.py
  wget -nc https://course.ccs.neu.edu/cs6120f25/assets/python/assignment7_unittests.py
  wget -nc https://course.ccs.neu.edu/cs6120f25/data/samsum/corpus.tar
  tar -xvf corpus.tar
  pip install dlai_grader
  ```

This set of files contains:

* **Your Code Template**: found in [`assignment7.py`](https://course.ccs.neu.edu/cs6120s25/assets/python/assignment7.py), which is as described above.

* **The SAMsum dataset**: around 16k paired conversations with their human-generated summaries, and can be found [here](https://course.ccs.neu.edu/cs6120s25/data/samsum/). Both the samples and their annotations are created by linguists, reflecting real-life messenger conversations: varying style, formality, slang, emojis, and general language patterns. 

* **Loading Scripts** found in [`utils.py`](https://course.ccs.neu.edu/cs6120s25/data/samsum/utils.py). The dataset has several functions that we will be using to process the data, including splitting training and test data from a folder name and preprocessing that data. We will be calling the majority of these utils functions from the `preprocess_data` function.

* **Unit Tests** found in [`assignment7_unittests.py`](https://course.ccs.neu.edu/cs6120s25/assets/python/assignment7_unittests.py). For every function that you're required to implement in [`assignment7.py`](https://course.ccs.neu.edu/cs6120s25/assets/python/assignment7.py), there is a corresponding `*_test()` function that does preliminary dimensionality checks, initial sanity checks, and some unit tests. This function calls out to the unit test file, which has additional parameter checks after tensorflow initialization. If you are using a different version of Tensorflow (than `v2.18.0`), then you may not necessarily get the same exact values, so you can either disregard the calls to `assignment7_unittest` or install `v2.18.0`.

Beyond the homework specific files, you can use the following resources. It is important to note that Transformers are compute heavy, and it is much easier to train if you're on GPU  hardware when training. You won't need to train until you've completely finished your Python coding, so you can save your Google credits until the end. There are a variety of ways that you can secure GPU's. To set resources, here are some options:

  * [Google Cloud Platform VMs](https://console.cloud.google.com/compute/): remember to shutdown / delete if using GPU's
  * [Google Cloud Workbench JupyterLab](https://console.cloud.google.com/vertex-ai/workbench): remember to shutdown / delete if using GPU's
  * [Google Colabs](https://colab.research.google.com/): remember to close your session if using the T4 GPU.

For any of the options above, you may find more luck in securing GPUs using the more expensive `NVIDIA L4` GPU Type. (Just remember to shut it off right after usage). You can check whether or not CUDA and Tensorflow are appropriately installed with `nvidia-smi` and `tf.config.list_physical_devices('GPU')`. In Colabs, your only available option is the `T4` GPU. If you hang on to the `T4` for too long, though, Google may shut you out of your account for a while so make sure you close your session. On GPU's, each iteration is around 20 seconds (except for the first iteration, which is 40 seconds). On CPU's, each iteration is around 20 minutes.

Note: all the unit tests in the homework were conducted with `tensorflow version 2.18.0`. There are minor differences in the tensorflow versions. You can check your version with:

```python
import tensorflow as tf
tf.__version__
```

You are free to *develop* in any environment (including [virtual machines](https://console.cloud.google.com/compute/instances) and [notebooks](https://console.cloud.google.com/vertex-ai/workbench)), but your submission must be a `*.py` file.

<br>
<br>

#### submission instructions

Submit your Python solution in [**assignment7.py**]({{ site.baseurl }}/assets/python/assignment7.py) to [Gradescope](http://gradescope.com).

