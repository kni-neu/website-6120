---
layout: page
title: homework 2
permalink: /homework-2/
description: neural network fundamentals
nav: true
hw_pdf: assignment-2.pdf
assigned: september 15
due_date: september 29
horizontal: false
---

<hr style="border:2px solid gray">
#### neural newtork fundamentals
-----

In this homework, we will explore the fundamentals of neural networks, implementing a two layer neural network by hand. An outsized portion of this homework (more than others) is rooted in linear algebra fundamentals, [Here are some mathematical properties](https://docs.google.com/presentation/d/1zy2veJEjDT-0acPbGsrEC93EP0MOZIx54jL-gA7wPqE) that will help you in derivations.  Review the homework in this [pdf file]({{ site.baseurl }}/assets/pdf/assignment-2.pdf). Remember that reading resources can be found in the [syllabus]({{ site.baseurl }}/syllabus).

<center>
<img 
  src="https://imageio.forbes.com/specials-images/imageserve/64f8e481ed69b0d89df9e2c7/Twitter-rebrands-to-X/960x0.png"
  width="500" height="auto">
</center>
<br>

-----
#### data and starter kit
-----

Along with some [helpful properties](https://docs.google.com/presentation/d/1zy2veJEjDT-0acPbGsrEC93EP0MOZIx54jL-gA7wPqE), we will use the [utility function](https://course.ccs.neu.edu/cs6120f25/data/twitter/utils.py) and [the twitter dataset](https://course.ccs.neu.edu/cs6120f25/data/twitter/twitter_data.pkl) available at our [homework 1 folder](https://course.ccs.neu.edu/cs6120f25/data/twitter/). You will also need the starter homework template [assignment2.py]({{ site.baseurl }}/assets/python/assignment2.py). Please develop in Python **without** the aid of libraries (e.g., tensorflow, keras, pytorch, jax, etc.) besides `numpy`, as our autograders will be grading accordingly. You are free to *develop* in any environment (including [virtual machines](https://console.cloud.google.com/compute/instances) and [notebooks](https://console.cloud.google.com/vertex-ai/workbench)), but your submission must be a `*.py` file.

Document templates can be either [Overleaf TeX File](https://www.overleaf.com/read/gbwryydmdjhv) or [DOCX File](https://docs.google.com/document/d/1Q8fpJo-gF_L0_TwUdw5E7x7faOAStK4n). When you've compiled/finished writing, **download the PDF** from Overleaf/Google and upload it to the submission link. 


-----
#### submission instructions
-----

Submit via [Gradescope](https://www.gradescope.com) before 5pm PT, Monday, September 29. Your artifacts will include:

* Compiled (or exported) PDF into a file called `assignment2.pdf`
* Data and parameters into a Python pickle file called `assignment2.pkl`
* All code with included functions in a file called `assignment2.py`
