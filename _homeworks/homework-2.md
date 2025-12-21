---
layout: page
title: homework 2
permalink: /homework-2/
description: neural network fundamentals
nav: true
hw_pdf: assignment-2.pdf
assigned: January 26, 2026
due_date: February 9, 2026
horizontal: false
---

<hr style="border:2px solid gray">
#### neural newtork fundamentals
-----

In this homework, we will explore the fundamentals of neural networks, implementing a two layer neural network by hand. An outsized portion of this homework (more than others) is rooted in foundational machine learning. Due to its difficulty, it will be due in two weeks.

<center>
<img 
  src="https://imageio.forbes.com/specials-images/imageserve/64f8e481ed69b0d89df9e2c7/Twitter-rebrands-to-X/960x0.png"
  width="500" height="auto">
</center>
<br>

-----
#### data and starter kit
-----

For this assignment, you must complete your work in the provided **starter template**: ➡️ [**assignment2.py**](https://course.ccs.neu.edu/cs6120s26/assets/python/assignment2.py). All of your code should go into this file. Submissions not based on this template may not run with the autograder. 

All utility and data files are available in the [homework 2's data folder](https://course.ccs.neu.edu/cs6120s26/data/twitter/). You will find:
- **Twitter dataset:** [twitter_data.pkl](https://course.ccs.neu.edu/cs6120s26/data/twitter/twitter_data.pkl)  
- **Utility functions:** [utils.py](https://course.ccs.neu.edu/cs6120s26/data/twitter/utils.py)  

Additionally, I've created some background reference slides about [helpful linear algebra and calculus properties](https://docs.google.com/presentation/d/1zy2veJEjDT-0acPbGsrEC93EP0MOZIx54jL-gA7wPqE).


-----
#### submission instructions
-----

Guidelines for development:
- Implement your solution in **Python**, using only `numpy`.  
- **Do not** use deep learning or auto-differentiation libraries (e.g., TensorFlow, PyTorch, Keras, JAX).  
- You may use either [LaTeX](https://www.overleaf.com/read/gbwryydmdjhv) or [Google Docs](https://docs.google.com/document/d/1Q8fpJo-gF_L0_TwUdw5E7x7faOAStK4n) templates for your writeup.

Submit via [Gradescope](https://www.gradescope.com) before 5pm PT, Monday, September 29. Your artifacts will include:

* Compiled (or exported) PDF into a file called `assignment2.pdf`
* Data and parameters into a Python pickle file called `assignment2.pkl`
* All code with included functions in a file called `assignment2.py`
