---
layout: page
title: homework 5
permalink: /homework-5/
description: word embeddings
nav: true
hw_pdf: assignment-5.pdf
assigned: october 20
due_date: november 10
horizontal: false
---

<hr style="border:2px solid gray">
#### creating word embeddings
-----

In this homework, you will be creating word embeddings from scratch by sampling. These embeddings derive from scientific papers and there are an example of specialized embeddings that we can later use for a variety of applications: including efficient searches of relevant papers (that don't necessarily use the exact same sets of words.). In later lectures, we will see that word vectors are often used as a fundamental component for downstream NLP tasks, e.g. question answering, text generation, translation, etc.. We will explore three types of word vectors: those derived from co-occurrence matrices, those derived from vanilla networks, and those derived from the famous algorithm *word2vec*.

**Note on Terminology**: The terms "word vectors" and "word embeddings" are often used interchangeably. The term "embedding" refers to the fact that we are encoding aspects of a word's meaning in a lower dimensional space. As Wikipedia states, "conceptually it involves a mathematical embedding from a space with one dimension per word to a continuous vector space with a much lower dimension".

Review the homework in this [pdf file]({{ site.baseurl }}/assets/pdf/assignment-5.pdf). Remember that reading resources can be found in the [syllabus]({{ site.baseurl }}/syllabus).

<center>
<img 
  src="https://upload.wikimedia.org/wikipedia/commons/6/6a/ArXiv-org_screenshot_20140706.png"
  width="700" height="auto">
</center>

-----
#### data and starter kit
-----

Your code template is available [here]({{ site.baseurl }}/assets/python/assignment5.py). We will be using data from [ArXiv](http://arxiv.org) today, containing the titles of over 3M academic and scientific papers. Per usual, you can find the datasets via the [course data site](https://course.ccs.neu.edu/cs6120s25/data/), where the dataset you can download is in the arxiv folder, titled [arxiv_titles.txt](https://course.ccs.neu.edu/cs6120s25/data/arxiv/arxiv_titles.txt). The data is formatted where each line is the title of a paper:

```
title-1
title-2
...
...
title-N
```

It is important to note that the papers are in sorted order (according to topic), and are not randomized in any way. There will _not_ be any `*_test` functions as your sampling and modeling results may vary. You may wish to write your own unit test functions with mock data.


<br>
#### submission instructions

* Prepare your `*.py` and PDF file with the requested functions, artifacts, and ensure that both are well-commented. Submission via [Gradescope](https://www.gradescope.com/courses/1042888) is before 5pm, Monday, November 10.

* For all three approaches, include in your writeup the nearest words for the following strings: "neural", "dark", "recurrent", "learning", "monaural", "recognition", "disparity", "expression", "retrieval", "genetic".

* Document templates can be either [Overleaf TeX File](https://www.overleaf.com/read/gbwryydmdjhv) or [DOCX File](https://docs.google.com/document/d/1Q8fpJo-gF_L0_TwUdw5E7x7faOAStK4n). When you've compiled/finished writing, **download the PDF** from Overleaf/Google and upload it to the submission link. 

* Make sure that you have documented your code with comments so that the TA can have an easier time understanding your logic. This will, in some cases, result in at least partial credit.

* We will be checking for plagiarism, comparing code that is too similar to classmate or past class alumni homework. This will automatically result in zero credit.


