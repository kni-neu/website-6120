---
layout: page
title: homework 3
permalink: /homework-3/
description: autocorrection in practice
nav: true
hw_pdf: assignment-3.pdf
assigned: september 29
due_date: october 6
horizontal: false
---

<hr style="border:2px solid gray">
#### the minimum edit distance
-----

In this homework, we will implement a functional and practical autocorrection and minimum edit distance algorithm. We will do so with dynamic programming algorithm determining the minimum edit distance. These models correct words that are 1 and 2 edit distances away. (We say two words are *n* edit distance away from each other when we need *n* edits to change one word into another.) Review the homework in this [pdf file]({{ site.baseurl }}/assets/pdf/assignment-3.pdf). Remember that reading resources can be found in the [syllabus]({{ site.baseurl }}/syllabus).

<center>
<img 
  src="../assets/img/misspelled.png"
  width="500" height="auto">
</center>
<br>

-----
#### data and starter kit
-----

You will need the [code](https://course.ccs.neu.edu/cs6120s25/assets/python/assignment3.py), corresponding [utility functions](https://course.ccs.neu.edu/cs6120s25/data/twitter/utils.py), and [shakespeare data](https://course.ccs.neu.edu/cs6120s25/data/shakespeare/shakespeare-edit.txt) at our [website](https://course.ccs.neu.edu/cs6120s25/data/shakespeare/). Please  develop in Python. 

You are free to *develop* in any environment (including [virtual machines](https://console.cloud.google.com/compute/instances) and [notebooks](https://console.cloud.google.com/vertex-ai/workbench)), but your submission must be a `*.py` file.

-----
#### submission instructions
-----

* Submit your Python file `assignment3.py` via  [Gradescope](https://www.gradescope.com) before 5pm, Monday, October 6.

<!-- * Document templates can be either [Overleaf TeX File](https://www.overleaf.com/read/gbwryydmdjhv) or [DOCX File](https://docs.google.com/document/d/1Q8fpJo-gF_L0_TwUdw5E7x7faOAStK4n). When you've compiled/finished writing, **download the PDF** from Overleaf/Google and upload it to the submission link.  -->



<!--
<br><br><br>
<hr style="border:2px solid gray">
#### project checkpoint
-----

Each week, there will be a checkpoint for your project so that you are on track to turn in the project at the end of the semester. This week

* start thinking about what types of topics you're interested in researching. Write a three of them down and explain what interests you about them.
-->
