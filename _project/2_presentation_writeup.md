---
layout: page
title: final llm demo 
permalink: /presentation-and-project/
description: final submission- demo and writeup
nav: true
nav_order: 2
hw_pdf: project-description.pdf
assigned: january 11, 2025
due_date: april 24, 2025
horizontal: false
---

<hr style="border:2px solid gray">

<br>

Welcome to the final project page. There are two options for our project: 

1. **submitting a paper to a prominent NLP venue.** For example, EMNLP, KDD, KDD Workshops, etc. Your delivery will be the PDF and your Github repository. While you do not need to create an endpoint, this option is difficult as you will need to replicate research, create comparisons to state of the art, and most importantly, make a novel contribution to the academic community.
2. **executing a demonstration for your project that leverages LLMs.** You will leverage LLMs and pair it with common natural language processing techniques for various applications (e.g., RAG). Below are two routes that you can choose for this project. This [laboratory](https://docs.google.com/presentation/d/1HcLInp203By38I5YYN5cABtHHh-p62zt) may be helpful for getting your server up and running.

If you are writing a paper, please ensure that your are working with teaching assistants and instructor very early on. It is likely that you will need to go through several ideation phases, proposals, and paper and data surveys throughout the experience. 

As many of you will likely be targeting the latter option of a demonstration, the remainder of this page is devoted to the project options that are available. The demonstration is considerably more closed ended, and you will choose one of two projects.

{% toc %}

<br>

-----
## demo project objectives

Feel free to pick any topic that you have data for. You can form groups of up to four; in your writeup, have a contributions sections denoting what each member worked on. There are three components to your project: your writeup, the demonstration, and your code. 

#### Create a RAG Inforamtion Retrieval System (Option A)

Retrieval Augmented Generation (RAG) is a framework for LLM powered systems that make use of external data sources. By doing so, RAG overcomes knowledge cutoff issues by leveraging external data sources to update the LLM's understanding of the world.

RAG serves as an alternative to re-training or fine-training models, which is expensive and difficult, and would often require repeated re-training to regularly update the model with new information. Recency is one known knowledge cutoff issue and relate to several applications (e.g., news outlets). Another knowledge cutoff would be exposure to specific or esoteric domain knowledge subject matter (e.g., medical documents, documents in particular scientific fields). It is a more flexible and less expensive way to overcome knowledge cutoffs, and is simple to implement as it pertains mostly to inference. 

For this project, *build a complete Retrieval Augmented Generation RAG system with your very own LLM.* Create a finished, deployable solution with a front end that leverages a model project with user interface, served on GCP. The following are some ideas of example projects:

#### Create Program-Aided Language Agent (Option B)

Program-Aided Language Models are those that allow your (LLM) model to interact with external applications that are good at some operation. For example, if you ask your LLM a math question, you could have your model engage with an application like a Python interpreter. The framework for augmenting LLMs in this way is called program-aided language models, or PAL for short. The method makes use of chain of thought prompting to generate executable scripts. The scripts that the model generates are passed to an interpreter / application to execute and solve the problem. 

For this project, *build a complete PAL Agent that will be able to take actions based on your interactions with it.* Using templated interactions / engagements actions based on interactions with an LLM, we can call to an application that is executable from your script. Ensure that guardrails are added for any actions that are undesirable. 

<br>

-----
## Starter Kit and Artifacts

In both paper and project, it is likely that we will need to ensure that we have sufficient GPU resources. Refer to the earlier labs in the class. The following are the artifacts that we will turn in via Gradescope.

* [Project Writeup](https://www.overleaf.com/read/xcjqmczwyrcz#0deb70) (PDF Format)
  - Click on `project_template.tex` to edit your project
  - Rreview the [Heilmeier Catechism](https://www.darpa.mil/work-with-us/heilmeier-catechism)?
* [Project Repository](http://www.github.com) (Github Repository via Link)
  - Include a README.md on how to setup and run the project. This should be straightforward.
  - Containerize your solution with Docker so we don't need specific libraries or software.
* [Demonstration](http://streamlit.io) (DNS unnecessary)
  - This will need to be up on presentation day. Ensure that we can access your server (which can be an IP address)

<br>

-----
## rubric and criterion

There are two components your project: its presentation and your technical delivery. This delivery will come in the form of a real-time demonstration of your engineering. In order to receive credit, your **must have a real-time inference** component. The delineation between the two contributions will be graded on the following.

|---|---|---|
|---|---|---|
| 33% | : | __Documentation__ : Written and oral delivery of project rationale and justification of approach
|     | - | &nbsp;&nbsp;&nbsp;&nbsp; Motivation and Impact: A future iteration of this project can make a difference
|     | - | &nbsp;&nbsp;&nbsp;&nbsp; Background and Related Work: Project is well-researched within and beyond course scope
|     | - | &nbsp;&nbsp;&nbsp;&nbsp; Modeling Methodology: Robustness to pitfalls like class imbalance and overfitting
|     | - | &nbsp;&nbsp;&nbsp;&nbsp; Evaluation and Analysis: Justification of the approach is sound
| 66% | : | __Technical__ : Demonstration of your work, handling of corner cases, technical correctness
|     | - | &nbsp;&nbsp;&nbsp;&nbsp; Base Accessibilty: The endpoint is open and is accessible to the public
|     | - | &nbsp;&nbsp;&nbsp;&nbsp; Performant: Results arrive in a reasonble time, and path to additional scaling is apparent
|     | - | &nbsp;&nbsp;&nbsp;&nbsp; Data Scale: Data needs to have at least 10k rows of meaningful amount of data
|     | - | &nbsp;&nbsp;&nbsp;&nbsp; Replicable: Algorithms must be replicable and containerized

<br>

-----
## presentation and grading

Each team will have five minutes to introduce their project (slides are not necessary). Prior to this, they will need start their servers so that the demonstration can commence. Instructor and TA's will operate the end-point in real-time from their laptops.

<br>

-----
## submission instructions
-----

Add the link to your project to [the final lecture slide deck](https://docs.google.com/presentation/d/1VO-SAmfm3smmDhVn6AkyK-SZTDURDILz9xCFx_0OiAA). (This is slide 2.) Commit all your code to your repository, and submit via [Gradescope](https://www.gradescope.com/). There, you will upload your PDF and provide the link to your repository, which should have all the code that you used to generate your solutions. Please submit only __once__ per team by including all persons on Gradescope.



<!--
<br><br><br>
<hr style="border:2px solid gray">
#### project checkpoint
-----
-->


