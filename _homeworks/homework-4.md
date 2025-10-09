---
layout: page
title: homework 4
permalink: /homework-4/
description: autocomplete with topical information
nav: true
hw_pdf: assignment-4.pdf
assigned: october 6
due_date: october 20
horizontal: false
---

<hr style="border:2px solid gray">
#### modeling short sequences of words
-----

Language models provide the capability to predict the most likely set of texts to follow any preceding text, of which the simplest model is the *n*-gram model. We will be building our first language model in this homework trained on Twitter data. Review the homework in this [pdf file]({{ site.baseurl }}/assets/pdf/assignment-4.pdf). Remember that reading resources can be found in the [syllabus]({{ site.baseurl }}/syllabus).

<center>
<img 
  src="https://assets.toptal.io/images?url=https%3A%2F%2Fbs-uploads.toptal.io%2Fblackfish-uploads%2Fcomponents%2Fblog_post_page%2F4085338%2Fcover_image%2Fregular_1708x683%2Fcover-0304-c32f070e8f972b73bb5c0c5404016669.png"
  width="500" height="auto">
</center>
<br>

-----
#### data and starter kit
-----

You will need [the data](https://course.ccs.neu.edu/cs6120f25/data/twitter/): the [`en_US.twitter.txt`](https://course.ccs.neu.edu/cs6120f25/data/twitter/en_US.twitter.txt) file and [the code]({{ site.baseurl }}/assets/python/assignment4.py). You can read more about this data [here](https://github.com/bquast/Data-Science-Capstone/blob/master/Online-Text-Exploration.md).

You will be filling out the portions in [the code]({{ site.baseurl }}/assets/python/assignment4.py) that say `<YOUR-CODE-HERE>`. There is also helpful unit test code with the suffix `_test()`. For example, 

  ```python
  def estimate_probabilities():
    """
    The graded function that you will need to fill out
    """
    # <YOUR-CODE-HERE>
    return None

  def estimate_probabilities_test():
    """
    Ungraded: You can use this function to test out estimate_probabilities. 
    """
    # Run this function to test 
    return
  ```

You are free to *develop* in any environment (including [virtual machines](https://console.cloud.google.com/compute/instances) and [notebooks](https://console.cloud.google.com/vertex-ai/workbench)), but your submission must be a `*.py` file.

<br>

#### submission instructions

* Submit your homework on [Gradescope, Assignment 4](https://www.gradescope.com/). You will need to upload your *well-commented* Python code (either as a notebook or as a Python file.)

