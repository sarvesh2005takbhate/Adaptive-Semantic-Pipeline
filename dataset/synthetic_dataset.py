"""
dataset_generator.py  —  Adaptive Semantic Parallelism
======================================================
Usage:
    python dataset_generator.py --num-samples 6000 --output data/dataset.jsonl
    python dataset_generator.py --num-samples 8000 --use-llm --output data/dataset.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.labels import (
    INTENT_LABELS,
    LABEL2ID,
    STRONG_INTENTS,
    WEAK_INTENTS,
    ROUTE_STRONG,
    ROUTE_WEAK,
)

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dataset_gen")

# we manually give templates for each intent, which are then filled with random variables to create a large variety of questions

TEMPLATES: Dict[str, List[str]] = {

    #  MATH (strong) 
    "math": [
        "Solve the equation {eq} for x",
        "Calculate the integral of {func} with respect to x",
        "Find the derivative of {func}",
        "Compute the sum of all {adj} integers from {a} to {b}",
        "Simplify the expression {expr}",
        "Prove that the sum of the first n natural numbers is n(n+1)/2",
        "Find the eigenvalues of the matrix [[{a},{b}],[{c},{d}]]",
        "Solve the system of equations: {eq1} and {eq2}",
        "Calculate the determinant of a {n}x{n} matrix",
        "Find the limit of {func} as x approaches {val}",
        "Evaluate the double integral of {func} over the region {region}",
        "Compute the Taylor series expansion of {func} around x={val}",
        "Find all roots of the polynomial {poly}",
        "Calculate the probability of {event} given {conditions}",
        "Solve the differential equation {ode}",
        "Find the area enclosed by the curves {curve1} and {curve2}",
        "Compute the Fourier transform of {func}",
        "Determine whether the series {series} converges or diverges",
        "Calculate the standard deviation of the dataset {data}",
        "Find the inverse of the matrix {matrix}",
        "Prove by induction that {statement}",
        "Compute the cross product of vectors {v1} and {v2}",
        "Solve the recurrence relation {recurrence}",
        "Find the maximum value of {func} subject to the constraint {constraint}",
        "Calculate the GCD and LCM of {a} and {b}",
        "Determine the number of permutations of {n} objects taken {r} at a time",
        "Find the equation of the tangent line to {curve} at point ({x},{y})",
        "Compute the Laplace transform of {func}",
        "Solve the optimization problem: minimize {func} subject to {constraints}",
        "Calculate the volume of the solid of revolution formed by rotating {curve} about the x-axis",
        "Find the partial derivatives of f(x,y) = {func}",
        "Determine whether the function {func} is continuous at x = {val}",
        "Calculate the arc length of {curve} from x={a} to x={b}",
        "Solve the inequality {ineq}",
        "Find the radius of convergence of the power series {series}",
        "Compute the gradient of f(x,y,z) = {func}",
        "Determine the rank of the matrix {matrix}",
        "Calculate the conditional probability P(A|B) given P(A)={pa}, P(B)={pb}, P(A∩B)={pab}",
        "Find the general solution of the second-order ODE {ode}",
        "Prove that √2 is irrational",
        "Calculate the expected value of the random variable X with distribution {dist}",
        "Find the surface area of the parametric surface {surface}",
        "Solve the linear programming problem: maximize {obj} subject to {constraints}",
        "Compute the divergence and curl of the vector field {field}",
        "Determine whether the matrix {matrix} is positive definite",
        "Find the Jordan normal form of the matrix {matrix}",
        "Calculate the moment of inertia of {shape} about {axis}",
        "Solve the Diophantine equation {eq}",
        "Find the least squares solution to the overdetermined system {system}",
        "Compute the convolution of {f1} and {f2}",
        "Determine the number of spanning trees of the graph {graph}",
        "Solve for x: log₂({expr}) = {val}",
        "Find the angle between the vectors {v1} and {v2}",
        "Calculate the work done by force F = {force} along the path {path}",
        "Evaluate the line integral of {func} along {curve}",
        "Compute the binomial coefficient C({n},{k})",
        "Find the area of the triangle with vertices ({x1},{y1}), ({x2},{y2}), ({x3},{y3})",
        "Solve the trigonometric equation {eq}",
        "Calculate the z-score for a value of {val} given mean={mean} and std={std}",
        "Find the centroid of the region bounded by {curves}",
        "Determine whether the vectors {v1}, {v2}, {v3} are linearly independent",
        "Compute the residue of {func} at the pole z = {z0}",
        "Solve the wave equation for {conditions}",
        "Calculate the correlation coefficient between {X} and {Y}",
        "Find all prime numbers between {a} and {b}",
        "Determine the number of ways to partition {n} into positive integers",
        "Solve the Cauchy-Euler equation {eq}",
        "Compute the Jacobian of the transformation {transform}",
        "Find the shortest path in the graph {graph} from node {s} to node {t}",
        "Calculate the variance of a binomial distribution with n={n} and p={p}",
        "Prove the Cauchy-Schwarz inequality",
        "Find the radius and center of the circle passing through {p1}, {p2}, {p3}",
        "Solve the heat equation for {conditions}",
        "Calculate the nth Fibonacci number using matrix exponentiation",
        "Find the volume of the tetrahedron with vertices {v1}, {v2}, {v3}, {v4}",
        "Determine whether {n} is a prime number and explain your method",
        "Compute the Hessian matrix of f(x,y) = {func}",
        "Solve the Bernoulli differential equation {eq}",
        "Find the inflection points of the function {func}",
        "Calculate the compound interest on principal {P} at rate {r} for {t} years",
    ],

    #  CODE (strong) 
    "code": [
        "Write a Python function that {task}",
        "Implement a {ds} data structure in {lang}",
        "Write code to {task} using {approach}",
        "Create a {lang} class that {behavior}",
        "Implement the {algo} algorithm in {lang}",
        "Write a function that takes {input_desc} and returns {output_desc}",
        "Debug the following code and fix all errors:\n{code_snippet}",
        "Write a recursive function to {task}",
        "Implement a REST API endpoint that {behavior}",
        "Write unit tests for a function that {behavior}",
        "Create a Python script that reads a CSV file and {task}",
        "Implement binary search in {lang}",
        "Write a function to validate {input_type}",
        "Create a decorator in Python that {behavior}",
        "Implement a thread-safe {ds} in {lang}",
        "Write a SQL query that {task}",
        "Implement merge sort in {lang} with O(n log n) time complexity",
        "Write a function that converts {format1} to {format2}",
        "Create a web scraper that extracts {data} from {site_type}",
        "Implement a cache with LRU eviction policy in {lang}",
        "Write a generator function in Python that yields {output}",
        "Implement the observer pattern in {lang}",
        "Write a function that finds all {pattern} in a string",
        "Create a command-line tool that {behavior}",
        "Implement a priority queue using a binary heap in {lang}",
        "Write code to parse {format} files and extract {fields}",
        "Implement depth-first search for a graph in {lang}",
        "Write a function that compresses a string using run-length encoding",
        "Create a rate limiter that allows {n} requests per {period}",
        "Implement the Singleton pattern in {lang}",
        "Write a function that merges two sorted arrays without extra space",
        "Create a simple state machine for {domain}",
        "Implement a trie data structure for autocomplete in {lang}",
        "Write a function to detect cycles in a linked list",
        "Create a middleware function that {behavior}",
        "Implement quicksort with three-way partitioning in {lang}",
        "Write a regex pattern that matches {pattern_desc}",
        "Create a function that deep clones a nested object in {lang}",
        "Implement the A* pathfinding algorithm in {lang}",
        "Write a function that generates all permutations of a list",
        "Create a connection pool for database connections in {lang}",
        "Implement Dijkstra's shortest path algorithm in {lang}",
        "Write a function that flattens a nested dictionary",
        "Create a pub/sub messaging system in {lang}",
        "Implement a balanced BST (AVL tree) in {lang}",
        "Write a function that evaluates a mathematical expression from a string",
        "Create a task scheduler with dependency resolution",
        "Implement consistent hashing in {lang}",
        "Write a function to find the longest common subsequence of two strings",
        "Create a simple garbage collector in {lang}",
        "Implement a bloom filter in {lang}",
        "Write a function that checks if a binary tree is balanced",
        "Create a circuit breaker pattern implementation in {lang}",
        "Implement topological sort for a directed acyclic graph",
        "Write a function that performs matrix multiplication in {lang}",
        "Create an event-driven architecture for {domain}",
        "Implement a skip list data structure in {lang}",
        "Write a function that finds the kth largest element in an unsorted array",
        "Create a simple load balancer with round-robin scheduling",
        "Implement the Knuth-Morris-Pratt string matching algorithm in {lang}",
        "Write a function that serializes and deserializes a binary tree",
        "Create an async task queue with retry logic in {lang}",
        "Implement a red-black tree in {lang}",
        "Write a function that finds all strongly connected components in a directed graph",
        "Create a simple ORM for SQLite in Python",
        "Implement a concurrent hash map in {lang}",
        "Write a function to solve the N-Queens problem",
        "Create a simple HTTP server from scratch in {lang}",
        "Implement Huffman coding for text compression in {lang}",
        "Write a function that computes the edit distance between two strings",
        "Create a thread pool implementation in {lang}",
        "Implement a B-tree data structure in {lang}",
        "Write a function that generates a maze and solves it",
        "Create a simple database index using B+ tree in {lang}",
        "Implement the Raft consensus algorithm in {lang}",
        "Write a function to convert infix expression to postfix",
        "Create a memory allocator in C",
        "Implement a minimum spanning tree algorithm in {lang}",
        "Write a function that implements the Fisher-Yates shuffle",
    ],

    #  SIMULATION (strong) 
    "simulation": [
        "Simulate {n} rounds of a {game} and report win probabilities",
        "Model the spread of {disease} in a population of {n} using the SIR model",
        "Run a Monte Carlo simulation to estimate the value of π using {n} random points",
        "Simulate a queuing system with {arrival} arrival rate and {service} service rate",
        "Model the trajectory of a projectile with initial velocity {v} at angle {theta}",
        "Simulate a random walk in {dim} dimensions for {steps} steps",
        "Model the evolution of a predator-prey system using Lotka-Volterra equations",
        "Simulate traffic flow on a {n}-lane highway with {density} vehicle density",
        "Run a simulation of {n} particles in a {type} system with {interaction} interactions",
        "Model the growth of a population with carrying capacity {K} and growth rate {r}",
        "Simulate an election with {n} candidates and {voters} voters under {system} voting",
        "Model the orbital mechanics of a two-body system with masses {m1} and {m2}",
        "Simulate a financial portfolio with {n} assets over {years} years",
        "Model heat diffusion in a {dim}-D rod with boundary conditions {bc}",
        "Simulate the evolution of a cellular automaton with rule {rule}",
        "Model the behavior of a spring-mass-damper system",
        "Simulate network packet routing with {n} nodes and {latency} latency",
        "Run an agent-based simulation of {scenario} with {n} agents",
        "Model the fluid dynamics of {fluid} flowing through a pipe of diameter {d}",
        "Simulate a supply chain with {n} suppliers and {demand} demand pattern",
        "Model radioactive decay of {element} with half-life {t}",
        "Simulate the double pendulum system and show chaotic behavior",
        "Model the spread of information in a social network of {n} nodes",
        "Simulate a genetic algorithm to optimize the {problem}",
        "Model the vibration of a string fixed at both ends",
        "Simulate a Markov chain with transition matrix {matrix}",
        "Model the interaction of {n} galaxies using N-body simulation",
        "Simulate queueing with multiple servers and priority classes",
        "Model the charging and discharging of a capacitor in an RC circuit",
        "Simulate the evolution of cooperation using the prisoner's dilemma with {n} agents",
        "Model weather patterns using a simplified atmospheric model",
        "Simulate the performance of a CPU cache with {policy} replacement policy",
        "Model the spread of a forest fire on a {n}x{n} grid",
        "Simulate a neural network learning XOR from scratch",
        "Model the dynamics of a three-body gravitational system",
        "Simulate an ecosystem with {n} species and {resources} shared resources",
        "Model the propagation of seismic waves through {medium}",
        "Simulate a distributed consensus protocol with {n} nodes and {f} failures",
        "Model the pharmacokinetics of drug {drug} with dose {dose}",
        "Simulate the Ising model on a {n}x{n} lattice at temperature {T}",
    ],

    #  RESEARCH (strong) 
    "research": [
        "Survey the recent literature on {topic} and summarize key findings",
        "Compare and contrast {method1} and {method2} for {task}",
        "What are the current state-of-the-art approaches to {problem}",
        "Analyze the strengths and weaknesses of {approach} for {application}",
        "Review the evolution of {field} from {year1} to {year2}",
        "What are the open research questions in {field}",
        "Critically evaluate the methodology used in {study_type} studies of {topic}",
        "Synthesize findings from multiple studies on the effect of {variable} on {outcome}",
        "Identify research gaps in the field of {field}",
        "How has {technology} impacted {domain} according to recent research",
        "What are the ethical implications of {technology} in {context}",
        "Compare quantitative and qualitative research methods for studying {topic}",
        "Analyze the reproducibility crisis in {field}",
        "What theoretical frameworks are used to study {phenomenon}",
        "Review the evidence for and against {hypothesis}",
        "How do different cultures approach {topic} according to cross-cultural research",
        "Examine the role of {factor} in {process} based on existing literature",
        "What are the methodological challenges in studying {topic}",
        "Synthesize the current understanding of {mechanism} in {system}",
        "Analyze how {policy} has affected {outcome} based on empirical evidence",
        "What are the key debates in {field} regarding {topic}",
        "Review the effectiveness of {intervention} for {condition}",
        "How has the definition of {concept} evolved in academic literature",
        "Identify the most influential papers in {field} from the last decade",
        "What are the limitations of current {method} approaches to {problem}",
        "Analyze the relationship between {variable1} and {variable2} across studies",
        "Review how {theory} has been applied in {domain}",
        "What are emerging trends in {field} research",
        "Compare the findings of longitudinal vs cross-sectional studies on {topic}",
        "Examine the environmental impact of {technology} based on life-cycle analyses",
        "What role does {factor} play in {outcome} according to meta-analyses",
        "Review the development of {metric} as a measure of {construct}",
        "Analyze the scalability challenges of {approach} in {domain}",
        "What are the best practices for {process} according to recent research",
        "Examine how {phenomenon} varies across different {context}",
        "Review the safety profile of {intervention} based on clinical trials",
        "What are the computational requirements of {method} compared to alternatives",
        "Analyze the societal implications of widespread {technology} adoption",
        "Review how machine learning has been applied to {domain}",
        "What are the key performance metrics used to evaluate {system_type} systems",
    ],

    #  PREDICTION (strong) 
    "prediction": [
        "Predict the trend of {metric} over the next {period} based on historical data",
        "Forecast the demand for {product} in {region} for the next {period}",
        "Estimate the future price of {asset} using {method}",
        "Predict the outcome of {event} given the current conditions",
        "What will be the impact of {change} on {metric} in the next {period}",
        "Forecast the population growth of {region} by {year}",
        "Predict the likelihood of {event} occurring within {timeframe}",
        "Estimate the failure probability of {system} given {conditions}",
        "Predict the customer churn rate for {period} based on {features}",
        "Forecast energy consumption for {building_type} in {season}",
        "Predict the effect of raising {parameter} by {amount} on {outcome}",
        "Estimate the time to completion for {project} given current progress",
        "Predict which {candidates} will {outcome} based on {data}",
        "Forecast the revenue growth if {strategy} is implemented",
        "Predict the optimal {parameter} to maximize {metric}",
        "Estimate the market size for {product} in {year}",
        "Predict the environmental impact of {policy} over {period}",
        "Forecast the spread of {phenomenon} using current trajectory data",
        "Predict the performance of {model} on {dataset} without running it",
        "Estimate the ROI of investing in {technology} over {period}",
        "Predict the bottleneck in {system} as load increases to {level}",
        "Forecast the adoption rate of {technology} in {market}",
        "Predict the weather for {location} in the next {days} days",
        "Estimate the computational cost of training {model} on {dataset}",
        "Predict the survival probability given {clinical_features}",
        "Forecast traffic congestion for {route} at {time}",
        "Predict the stock market sector rotation for the next quarter",
        "Estimate the required sample size to detect effect size {d} with power {power}",
        "Predict the winner of the {competition} based on historical performance",
        "Forecast the inflation rate for {country} in the next {period}",
        "Predict the customer lifetime value for segment {segment}",
        "Estimate the probability of a {disaster} in {region} within {period}",
        "Predict the impact of {feature} on user retention",
        "Forecast the growth of {cryptocurrency} market cap in {period}",
        "Predict which features will most improve {metric} for {system}",
        "Estimate the economic cost of {event} if it occurs",
        "Predict the optimal pricing for {product} to maximize revenue",
        "Forecast the skill gap in {industry} for the next {years} years",
        "Predict the outcome of A/B test between {variant_a} and {variant_b}",
        "Estimate the carbon footprint reduction from {intervention}",
    ],

    #  DATA_ANALYSIS (strong) 
    "data_analysis": [
        "Analyze the correlation between {var1} and {var2} in the dataset",
        "Perform exploratory data analysis on the {dataset} dataset",
        "Identify outliers in the {column} column using {method}",
        "Calculate descriptive statistics for {variable} grouped by {category}",
        "Classify {items} into {n} categories based on {features}",
        "Cluster the data using {algorithm} and interpret the results",
        "Build a regression model to predict {target} from {features}",
        "Perform a hypothesis test to determine if {claim}",
        "Analyze the sentiment of {text_source} reviews",
        "Create a pivot table showing {metric} by {dim1} and {dim2}",
        "Detect anomalies in the time series data for {metric}",
        "Perform feature importance analysis for predicting {target}",
        "Calculate the chi-square statistic for the contingency table of {var1} vs {var2}",
        "Normalize the {dataset} dataset and explain the transformation",
        "Analyze the distribution of {variable} and test for normality",
        "Perform principal component analysis on the {dataset} dataset",
        "Calculate the moving average of {metric} over {window} periods",
        "Segment customers based on their {behavior} data",
        "Analyze the A/B test results for {experiment}",
        "Calculate the confidence interval for the mean of {variable}",
        "Perform survival analysis on the {dataset} dataset",
        "Analyze the text frequency distribution in {corpus}",
        "Build a confusion matrix for the {model} predictions",
        "Perform time series decomposition of {metric} into trend, seasonality, and residuals",
        "Calculate the Gini coefficient for the income distribution in {dataset}",
        "Analyze the network structure of {graph} and find communities",
        "Perform ANOVA to compare {groups} on {metric}",
        "Calculate the ROC curve and AUC for the {classifier}",
        "Analyze the spatial distribution of {events} in {region}",
        "Perform association rule mining on the {transaction} dataset",
        "Calculate the PageRank scores for nodes in the {network}",
        "Analyze the variance inflation factors for {features}",
        "Perform Granger causality test between {series1} and {series2}",
        "Build a decision tree to classify {target} and visualize it",
        "Analyze the word embeddings of {terms} and find analogies",
        "Perform dimensionality reduction on {dataset} using t-SNE",
        "Calculate the mutual information between {var1} and {var2}",
        "Analyze the seasonal patterns in {metric} data",
        "Perform cross-validation on {model} with {k} folds",
        "Calculate the effect size (Cohen's d) between {group1} and {group2}",
    ],

    #   TRANSLATION (weak) 
    "translation": [
        "Translate the following text to {lang}: \"{text}\"",
        "Convert this {src_lang} sentence to {tgt_lang}: \"{text}\"",
        "Translate the technical documentation from {src_lang} to {tgt_lang}",
        "How do you say \"{phrase}\" in {lang}",
        "Translate this medical term to {lang}: {term}",
        "Convert the following code comments from {src_lang} to {tgt_lang}",
        "Translate this legal clause to {lang} preserving formal tone",
        "Translate \"{text}\" to {lang} using informal register",
        "Provide the {lang} translation of these UI strings: {strings}",
        "Translate this marketing slogan to {lang}: \"{slogan}\"",
        "Convert the following proverb to its {lang} equivalent: \"{proverb}\"",
        "Translate this error message to {lang}: \"{error}\"",
        "Translate the following menu items to {lang}: {items}",
        "Convert this academic abstract to {lang}",
        "Translate the following email from {src_lang} to {tgt_lang}",
        "How would you express \"{idiom}\" in {lang}",
        "Translate these API endpoint descriptions to {lang}",
        "Convert the following warning label to {lang}: \"{text}\"",
        "Translate this poem to {lang} preserving the rhyme scheme",
        "Translate the user interface text to {lang}: {ui_text}",
        "Convert this recipe from {src_lang} to {tgt_lang}",
        "Translate the following interview transcript to {lang}",
        "Translate these product descriptions to {lang}",
        "Convert the following safety instructions to {lang}",
        "Translate this business proposal to {lang}",
        "Translate the FAQ section to {lang}",
        "Convert this technical specification to {lang}",
        "Translate the following song title and description to {lang}",
        "Translate these database field names to {lang}",
        "Convert the patient information leaflet to {lang}",
        "Translate the following social media post to {lang}: \"{text}\"",
        "Translate the terms of service to {lang}",
        "Convert the following news headline to {lang}",
        "Translate this children's story to {lang}",
        "Translate the cooking instructions to {lang}",
        "Convert the weather report to {lang}",
        "Translate the travel itinerary to {lang}",
        "Translate these app store descriptions to {lang}",
        "Convert the museum exhibit descriptions to {lang}",
        "Translate the following customer review to {lang}: \"{text}\"",
    ],

    #  SUMMARIZATION (weak) 
    "summarization": [
        "Summarize the following article in {n} sentences: {text}",
        "Provide a brief summary of the key points in this document",
        "Write a TL;DR for the following text: {text}",
        "Summarize the main arguments presented in this essay",
        "Create an executive summary of this report",
        "Condense the following {n}-page document into a one-page summary",
        "Summarize the findings of this research paper",
        "Write a brief overview of the following meeting notes",
        "Summarize the plot of this story in {n} sentences",
        "Create a bullet-point summary of the key takeaways",
        "Summarize the pros and cons discussed in this review",
        "Write a one-paragraph summary of this chapter",
        "Summarize the main conclusions of this study",
        "Provide a high-level summary of this technical documentation",
        "Summarize the timeline of events described in this article",
        "Write a 100-word summary of this book chapter",
        "Summarize the patient's medical history from these notes",
        "Create an abstract for this paper based on its content",
        "Summarize the key financial metrics from this quarterly report",
        "Write a summary of the debate between {person1} and {person2}",
        "Summarize the product features listed in this specification",
        "Create a summary of the changes in this software release",
        "Summarize the legal arguments presented in this case",
        "Write a brief recap of the project status update",
        "Summarize the customer feedback from these survey responses",
        "Create a one-line summary for each section of this document",
        "Summarize the dietary guidelines from this health article",
        "Write a summary of the historical events covered in this chapter",
        "Summarize the policy changes described in this announcement",
        "Create a summary comparing the {n} proposals presented",
        "Summarize the interview highlights from this transcript",
        "Write a concise summary of the safety procedures",
        "Summarize the workflow described in this process document",
        "Create a summary of the test results from this report",
        "Summarize the recommendations made in this advisory",
        "Write a brief summary of the course syllabus",
        "Summarize the risk factors identified in this assessment",
        "Create a summary of the user research findings",
        "Summarize the competitive analysis from this market report",
        "Write a summary of the key decisions from this board meeting",
    ],

    #  EXPLANATION (weak) 
    "explanation": [
        "Explain how {concept} works in simple terms",
        "What is the difference between {thing1} and {thing2}",
        "Describe the process of {process}",
        "Explain why {phenomenon} occurs",
        "How does {technology} work under the hood",
        "Explain the concept of {concept} to a {audience}",
        "What are the main causes of {event}",
        "Describe the relationship between {concept1} and {concept2}",
        "Explain the significance of {event} in {field}",
        "How does {system} handle {situation}",
        "Explain the trade-offs between {option1} and {option2}",
        "What happens when {condition} occurs in {system}",
        "Describe the architecture of {system}",
        "Explain the role of {component} in {system}",
        "How is {metric} calculated and what does it measure",
        "Explain the advantages of {approach} over {alternative}",
        "What is {term} and why is it important in {field}",
        "Describe the lifecycle of {entity} in {context}",
        "Explain how {algorithm} achieves {property}",
        "What are the key principles behind {framework}",
        "Explain the historical context of {concept}",
        "How does {protocol} ensure {property}",
        "Describe the difference between {type1} and {type2} {things}",
        "Explain the mechanism by which {process} occurs",
        "What factors influence {outcome} in {context}",
        "Explain the concept of {concept} using an analogy",
        "How does the {organ} function in the human body",
        "Describe the steps involved in {process}",
        "Explain why {method} is preferred over {alternative} for {task}",
        "What are the implications of {finding} for {field}",
        "Explain how {feature} improves {metric} in {system}",
        "Describe the evolution of {concept} from {era1} to {era2}",
        "Explain the mathematical intuition behind {concept}",
        "How does {technique} prevent {problem}",
        "Describe the failure modes of {system}",
        "Explain the connection between {field1} and {field2}",
        "What is the purpose of {component} in {system}",
        "Explain how {policy} affects {stakeholder}",
        "Describe the security model of {system}",
        "Explain the fundamentals of {topic} for beginners",
    ],

    #  COMMUNICATION (weak) 
    "communication": [
        "Draft a professional email to {recipient} about {topic}",
        "Write a polite follow-up message regarding {topic}",
        "Compose a formal letter requesting {request}",
        "Draft a meeting invitation for {topic} on {date}",
        "Write a thank-you note to {person} for {reason}",
        "Compose an apology email to {recipient} for {issue}",
        "Draft a project status update email for {stakeholders}",
        "Write a rejection letter that is professional and empathetic",
        "Compose a LinkedIn message introducing yourself to {person}",
        "Draft an announcement about {news} for the team",
        "Write a customer service response to a complaint about {issue}",
        "Compose a cover letter for a {position} role at {company}",
        "Draft a performance review feedback for {employee}",
        "Write a proposal email pitching {idea} to {audience}",
        "Compose a farewell message to your team",
        "Draft a cold outreach email to {prospect} about {product}",
        "Write a recommendation letter for {person} for {purpose}",
        "Compose a newsletter update about {topic}",
        "Draft a negotiation email regarding {terms}",
        "Write a conflict resolution message to {party}",
        "Compose an invitation to speak at {event}",
        "Draft an escalation email about {issue} to {manager}",
        "Write a partnership proposal to {organization}",
        "Compose a reminder email about {deadline}",
        "Draft a congratulations message for {achievement}",
        "Write a feedback request email to {recipients}",
        "Compose a change management communication about {change}",
        "Draft an out-of-office message for {period}",
        "Write a request for information about {topic}",
        "Compose a team motivation message during {situation}",
        "Draft a client onboarding welcome email",
        "Write a bug report email to the development team about {bug}",
        "Compose a sponsorship request for {event}",
        "Draft a meeting summary email with action items",
        "Write a volunteer recruitment message for {cause}",
        "Compose a press release about {announcement}",
        "Draft an internal memo about {policy_change}",
        "Write a donor appreciation letter",
        "Compose a product launch announcement email",
        "Draft a salary negotiation email",
    ],

    #  DOCUMENTATION (weak) 
    "documentation": [
        "Write a README file for a {project_type} project",
        "Create API documentation for the {endpoint} endpoint",
        "Document the installation steps for {software}",
        "Write a user guide for {feature}",
        "Create a troubleshooting guide for common {system} issues",
        "Document the database schema for {system}",
        "Write a changelog entry for version {version}",
        "Create a developer onboarding guide for {project}",
        "Document the deployment process for {service}",
        "Write a configuration guide for {tool}",
        "Create a code style guide for {language} projects",
        "Document the architecture decisions for {system}",
        "Write a migration guide from {old_version} to {new_version}",
        "Create an FAQ document for {product}",
        "Document the testing strategy for {project}",
        "Write a security best practices guide for {technology}",
        "Create a runbook for {operation}",
        "Document the CI/CD pipeline for {project}",
        "Write a data dictionary for {database}",
        "Create a contributing guide for the open-source {project}",
        "Document the error codes and their meanings for {system}",
        "Write a performance tuning guide for {system}",
        "Create a disaster recovery plan for {service}",
        "Document the authentication flow for {application}",
        "Write a release notes template for {product}",
        "Create a glossary of terms for {domain}",
        "Document the monitoring and alerting setup for {system}",
        "Write a compliance documentation for {standard}",
        "Create a knowledge base article about {topic}",
        "Document the backup and restore procedures for {system}",
        "Write an incident response playbook for {scenario}",
        "Create a capacity planning document for {service}",
        "Document the feature flag system for {project}",
        "Write an SLA document for {service}",
        "Create a network architecture document for {infrastructure}",
        "Document the logging conventions for {application}",
        "Write a data governance policy document",
        "Create a service catalog for {organization}",
        "Document the API versioning strategy for {project}",
        "Write a technical design document for {feature}",
    ],
}

HARD_ROUTING = [
    ("write a simple hello world function", "code", 0.2, "weak_model"),
    ("explain deep learning transformer architecture in detail", "explanation", 0.9, "strong_model"),
    ("translate hello to french", "translation", 0.2, "weak_model"),
    ("translate complex legal document to french with explanation", "translation", 0.9, "strong_model"),
]

def make_natural(text, rng):
    replacements = {
        "calculate": ["figure out", "work out", "compute"],
        "translate": ["convert", "rewrite in another language"],
        "summarize": ["make this shorter", "give a quick version"],
        "analyze": ["look into", "check what's going on"],
        "simulate": ["model", "try to mimic"],
    }

    for k, vals in replacements.items():
        if k in text.lower():
            text = text.replace(k, rng.choice(vals))

    return text

FILL_BANK: Dict[str, List[str]] = {
    "eq":       ["2x² + 3x - 5 = 0", "3x + 7 = 22", "x³ - 6x² + 11x - 6 = 0", "sin(x) = 0.5", "e^x = 10"],
    "func":     ["sin(x)/x", "x²·ln(x)", "e^(-x²)", "1/(1+x²)", "x·cos(x)", "√(x²+1)"],
    "expr":     ["(3x+2)(x-4)", "(a+b)³", "log(x²y)", "sin²(x)+cos²(x)-1"],
    "adj":      ["even", "odd", "prime", "positive"],
    "a":        ["1", "10", "1", "100"],
    "b":        ["50", "100", "1000", "500"],
    "c":        ["3", "-1", "0", "7"],
    "d":        ["4", "2", "5", "-3"],
    "n":        ["3", "4", "5", "10", "100"],
    "k":        ["2", "3", "5"],
    "val":      ["0", "1", "infinity", "π", "2"],
    "eq1":      ["2x + 3y = 7", "x - y = 1"],
    "eq2":      ["x + y = 5", "3x + 2y = 12"],
    "poly":     ["x³ - 3x + 2", "x⁴ - 1", "2x² - 7x + 3"],
    "region":   ["[0,1]×[0,1]", "the unit circle", "the first quadrant"],
    "curve":    ["y = x²", "y = sin(x)", "y = e^x"],
    "curve1":   ["y = x²", "y = 2x"],
    "curve2":   ["y = x", "y = x³"],
    "series":   ["Σ 1/n²", "Σ (-1)^n/n", "Σ n!/nⁿ"],
    "data":     ["{4, 8, 15, 16, 23, 42}", "{1, 2, 3, 4, 5}"],
    "matrix":   ["[[1,2],[3,4]]", "[[2,1,0],[1,3,1],[0,1,2]]"],
    "v1":       ["(1,2,3)", "(3,-1,4)"],
    "v2":       ["(4,5,6)", "(2,5,-1)"],
    "v3":       ["(7,8,9)", "(0,1,3)"],
    "recurrence":["a(n) = 2a(n-1) + 1", "f(n) = f(n-1) + f(n-2)"],
    "constraint":["x + y ≤ 10", "x² + y² = 1"],
    "constraints":["x ≥ 0, y ≥ 0, x+y ≤ 10", "Ax ≤ b, x ≥ 0"],
    "ineq":     ["2x - 3 > 5", "|x+1| < 4"],
    "ode":      ["y'' + 4y = 0", "dy/dx = xy", "y' - 2y = e^x"],
    "statement":["2ⁿ > n for all n ≥ 1", "n³ - n is divisible by 6"],
    "field":    ["F = (xy, yz, xz)", "F = (x², y², z²)"],
    "x":        ["1", "0", "π"],
    "y":        ["1", "2"],
    "x1": ["0"], "y1": ["0"], "x2": ["3"], "y2": ["4"], "x3": ["6"], "y3": ["0"],
    "p1": ["(0,0)"], "p2": ["(1,0)"], "p3": ["(0,1)"],
    "v4":       ["(1,0,0,1)"],
    "pa": ["0.3"], "pb": ["0.5"], "pab": ["0.1"],
    "mean": ["100"], "std": ["15"],
    "dist":     ["Poisson(λ=3)", "Normal(0,1)", "Binomial(10,0.5)"],
    "surface":  ["the sphere of radius R", "the torus"],
    "obj":      ["3x + 5y", "2x₁ + x₂"],
    "transform":["(u,v) → (u²-v², 2uv)"],
    "graph":    ["K₅", "the Petersen graph", "a 4-regular graph on 8 nodes"],
    "system":   ["Ax = b where A is 5×3"],
    "f1": ["e^(-x²)"], "f2": ["sin(x)"],
    "force":    ["(x, 2y, z)"],
    "path":     ["from (0,0,0) to (1,1,1)"],
    "r":        ["0.05", "0.08"],
    "t":        ["10", "5"],
    "z0":       ["0", "i"],
    "X": ["height"], "Y": ["weight"],
    "p":        ["0.5", "0.3"],
    "P":        ["1000", "5000"],
    "event":    ["rolling a 6 twice", "drawing an ace"],
    "conditions":["a fair die", "no replacement"],

    "task":     [
        "sorts a list of dictionaries by a given key",
        "finds the longest palindromic substring",
        "validates email addresses using regex",
        "converts Roman numerals to integers",
        "implements a basic calculator",
        "finds all anagrams of a word in a list",
        "counts word frequency in a text file",
        "checks if a string is a valid IPv4 address",
        "flattens a nested list of arbitrary depth",
        "removes duplicate elements while preserving order",
    ],
    "ds":       ["linked list", "hash map", "binary heap", "graph", "stack", "queue", "trie"],
    "lang":     ["Python", "Java", "C++", "Go", "Rust", "TypeScript", "JavaScript"],
    "approach": ["dynamic programming", "recursion", "iteration", "greedy algorithm", "BFS"],
    "behavior": ["handles concurrent requests safely", "retries failed operations", "validates input data", "logs all operations"],
    "algo":     ["quicksort", "Dijkstra's", "BFS", "DFS", "Kruskal's", "Floyd-Warshall"],
    "input_desc": ["a list of integers", "a string and a pattern", "a 2D matrix"],
    "output_desc": ["the sorted list", "all matching indices", "the maximum sum path"],
    "code_snippet": ["def fib(n): return fib(n-1) + fib(n-2)", "for i in range(len(lst)): if lst[i] = target: return i"],
    "input_type": ["email addresses", "phone numbers", "JSON payloads", "URLs"],
    "format1":  ["JSON", "XML", "YAML", "CSV"],
    "format2":  ["CSV", "JSON", "YAML", "Markdown"],
    "data":     ["{4, 8, 15, 16, 23, 42}"],
    "site_type":["news websites", "e-commerce sites", "job boards"],
    "output":   ["Fibonacci numbers", "prime numbers", "date ranges"],
    "pattern":  ["palindromes", "anagrams", "duplicates"],
    "domain":   ["e-commerce order processing", "user authentication", "payment processing", "file uploads"],
    "format":   ["JSON", "XML", "YAML", "CSV"],
    "fields":   ["name, email, phone", "id, timestamp, value", "title, author, date"],
    "period":   ["second", "minute", "hour"],
    "pattern_desc": ["valid email addresses", "ISO date strings", "hex color codes"],

    "game":     ["coin flip", "dice roll", "rock-paper-scissors", "blackjack"],
    "disease":  ["influenza", "COVID-19", "measles"],
    "arrival":  ["Poisson(λ=5)", "uniform"],
    "service":  ["exponential(μ=6)", "constant(2s)"],
    "v":        ["50 m/s", "100 m/s"],
    "theta":    ["30°", "45°", "60°"],
    "dim":      ["2", "3"],
    "steps":    ["1000", "10000"],
    "type":     ["gravitational", "electromagnetic"],
    "interaction": ["pairwise", "nearest-neighbor"],
    "K":        ["10000", "1000000"],
    "voters":   ["10000", "1000000"],
    "years":    ["10", "20", "30"],
    "bc":       ["T(0)=100, T(L)=0", "insulated ends"],
    "rule":     ["30", "110", "90"],
    "density":  ["0.3", "0.7"],
    "scenario": ["market competition", "evacuation", "resource sharing"],
    "fluid":    ["water", "air", "oil"],
    "element":  ["Carbon-14", "Uranium-238"],
    "medium":   ["rock", "water", "air"],
    "drug":     ["aspirin", "ibuprofen"],
    "dose":     ["500mg", "200mg"],
    "T":        ["2.0", "2.27", "3.0"],
    "policy":   ["LRU", "FIFO", "LFU"],
    "f":        ["1", "2"],
    "resources":["3", "5"],

    "topic":    [
        "transformer architectures in NLP",
        "climate change adaptation strategies",
        "quantum computing error correction",
        "CRISPR gene editing applications",
        "autonomous vehicle safety",
        "antibiotic resistance mechanisms",
        "renewable energy storage solutions",
        "federated learning privacy",
    ],
    "method1":  ["CNNs", "gradient boosting", "random forests", "SVMs"],
    "method2":  ["transformers", "neural networks", "Bayesian methods", "ensemble methods"],
    "problem":  ["natural language understanding", "protein folding", "image segmentation"],
    "application": ["medical imaging", "autonomous driving", "fraud detection"],
    "field":    ["machine learning", "genomics", "climate science", "neuroscience", "economics"],
    "year1":    ["2000", "2010"],
    "year2":    ["2020", "2024"],
    "study_type": ["observational", "randomized controlled", "cohort"],
    "variable": ["temperature", "dosage", "learning rate", "sample size"],
    "variable1":["education level", "sleep duration", "exercise frequency"],
    "variable2":["income", "cognitive performance", "cardiovascular health"],
    "outcome":  ["patient outcomes", "model accuracy", "crop yield"],
    "technology":["blockchain", "CRISPR", "LLMs", "IoT", "5G"],
    "context":  ["healthcare", "education", "finance", "agriculture"],
    "hypothesis":["the hygiene hypothesis", "efficient market hypothesis"],
    "phenomenon":["neuroplasticity", "the greenhouse effect", "quantum entanglement"],
    "factor":   ["socioeconomic status", "data quality", "model complexity"],
    "process":  ["protein synthesis", "natural selection", "machine learning training"],
    "intervention":["cognitive behavioral therapy", "vaccine boosters"],
    "condition":["depression", "diabetes", "Alzheimer's"],
    "concept":  ["entropy", "overfitting", "recursion", "supply and demand"],
    "metric":   ["F1 score", "BLEU", "accuracy", "latency"],
    "construct":["intelligence", "job satisfaction", "code quality"],
    "theory":   ["information theory", "game theory", "complexity theory"],
    "system_type": ["recommendation", "retrieval", "classification"],

    "product":  ["electric vehicles", "smartphones", "cloud services"],
    "region":   ["North America", "Southeast Asia", "Europe"],
    "asset":    ["gold", "Bitcoin", "S&P 500"],
    "method":   ["ARIMA", "LSTM", "Prophet"],
    "change":   ["increasing interest rates", "new regulations"],
    "year":     ["2030", "2025", "2035"],
    "timeframe":["6 months", "1 year", "5 years"],
    "features": ["purchase history, demographics, engagement", "age, income, location"],
    "building_type": ["commercial offices", "residential buildings", "data centers"],
    "season":   ["summer", "winter"],
    "parameter":["the learning rate", "the tax rate", "the dosage"],
    "amount":   ["10%", "50%", "2x"],
    "project":  ["the new feature rollout", "the migration project"],
    "candidates":["models", "strategies", "configurations"],
    "strategy": ["aggressive marketing", "price reduction"],
    "model":    ["GPT-4", "ResNet-50", "XGBoost"],
    "dataset":  ["ImageNet", "CIFAR-10", "the company dataset"],
    "level":    ["10x", "100x"],
    "market":   ["US", "global", "emerging markets"],
    "location": ["New York", "London", "Tokyo"],
    "days":     ["3", "5", "7"],
    "competition": ["the tournament", "the hackathon"],
    "country":  ["US", "UK", "India", "Japan"],
    "segment":  ["premium users", "free-tier users"],
    "disaster": ["flood", "earthquake", "hurricane"],
    "feature":  ["dark mode", "notifications", "social sharing"],
    "cryptocurrency": ["Bitcoin", "Ethereum"],
    "industry": ["AI/ML", "cybersecurity", "healthcare"],
    "variant_a":["the new design", "Plan A"],
    "variant_b":["the current design", "Plan B"],
    "clinical_features": ["age, stage, biomarkers"],
    "route":    ["Highway 101", "the downtown corridor"],
    "time":     ["8 AM", "5 PM"],
    "power":    ["0.8", "0.9"],

    "var1":     ["age", "income", "temperature"],
    "var2":     ["purchase_amount", "satisfaction", "productivity"],
    "column":   ["revenue", "response_time", "temperature"],
    "algorithm":["K-means", "DBSCAN", "hierarchical clustering"],
    "target":   ["churn", "price", "survival"],
    "category": ["region", "age_group", "product_type"],
    "items":    ["customer reviews", "support tickets", "news articles"],
    "text_source": ["Amazon product", "Yelp restaurant", "app store"],
    "dim1":     ["month", "region"],
    "dim2":     ["product_category", "customer_segment"],
    "window":   ["7-day", "30-day", "90-day"],
    "experiment":["the checkout redesign", "the pricing experiment"],
    "corpus":   ["the news corpus", "the academic papers", "the social media posts"],
    "classifier":["logistic regression", "random forest", "SVM"],
    "groups":   ["treatment A, treatment B, and control", "3 product versions"],
    "events":   ["crime incidents", "traffic accidents", "disease cases"],
    "series1":  ["oil prices", "GDP growth"],
    "series2":  ["stock returns", "unemployment"],
    "terms":    ["king, queen, man, woman", "Python, Java, code, program"],
    "group1":   ["treatment", "experimental"],
    "group2":   ["control", "baseline"],
    "network":  ["the citation network", "the social graph"],

    "lang":     ["French", "Spanish", "German", "Japanese", "Mandarin", "Arabic", "Hindi", "Korean", "Portuguese", "Russian"],
    "src_lang": ["English", "French", "Spanish", "German"],
    "tgt_lang": ["French", "English", "Japanese", "Mandarin"],
    "text":     [
        "The quick brown fox jumps over the lazy dog",
        "Please confirm your appointment for tomorrow",
        "Your order has been shipped and will arrive in 3-5 days",
        "The meeting has been rescheduled to next Monday",
        "Warning: This product contains allergens",
    ],
    "phrase":   ["How are you doing today", "Thank you for your help", "I don't understand"],
    "term":     ["myocardial infarction", "hypertension", "bilateral pneumonia"],
    "strings":  ["Submit, Cancel, Save, Delete, Edit", "Login, Signup, Forgot Password"],
    "slogan":   ["Think Different", "Just Do It", "Innovation for Everyone"],
    "proverb":  ["A stitch in time saves nine", "The early bird catches the worm"],
    "error":    ["Connection timed out. Please try again.", "Invalid credentials"],
    "items":    ["Appetizers, Main Course, Desserts, Beverages"],
    "idiom":    ["Break a leg", "Bite the bullet", "Piece of cake"],
    "ui_text":  ["Settings, Profile, Notifications, Help, Sign Out"],

    "person1":  ["the CEO", "Dr. Smith"],
    "person2":  ["the board", "Prof. Jones"],

    "thing1":   ["TCP", "supervised learning", "REST", "SQL", "a stack"],
    "thing2":   ["UDP", "unsupervised learning", "GraphQL", "NoSQL", "a queue"],
    "audience": ["a 5-year-old", "a non-technical manager", "a college student"],
    "organ":    ["heart", "liver", "brain", "kidney"],
    "era1":     ["classical", "pre-internet", "1990s"],
    "era2":     ["modern", "post-internet", "today"],
    "option1":  ["consistency", "speed", "simplicity"],
    "option2":  ["availability", "accuracy", "flexibility"],
    "component":["the garbage collector", "the scheduler", "the load balancer"],
    "system":   ["Linux", "Kubernetes", "TCP/IP", "Git"],
    "protocol": ["HTTPS", "OAuth 2.0", "WebSocket"],
    "property": ["data integrity", "confidentiality", "fault tolerance"],
    "type1":    ["relational", "compiled", "synchronous"],
    "type2":    ["non-relational", "interpreted", "asynchronous"],
    "things":   ["databases", "programming languages", "APIs"],
    "technique":["rate limiting", "caching", "sharding"],
    "finding":  ["AI bias", "climate feedback loops"],
    "situation":["network partition", "high load", "disk failure"],
    "stakeholder": ["small businesses", "students", "patients"],
    "field1":   ["mathematics", "biology", "economics"],
    "field2":   ["computer science", "medicine", "psychology"],

    "recipient":["the hiring manager", "a client", "the support team", "your professor"],
    "person":   ["a colleague", "your mentor", "a client"],
    "reason":   ["their mentorship", "attending the event", "the referral"],
    "date":     ["next Monday at 2 PM", "March 15th at 10 AM"],
    "issue":    ["the delayed delivery", "the billing error", "the downtime"],
    "stakeholders": ["senior management", "the engineering team", "all departments"],
    "position": ["software engineer", "data scientist", "product manager"],
    "company":  ["a tech startup", "a Fortune 500 company"],
    "employee": ["a junior developer", "a team lead"],
    "idea":     ["a new product feature", "a cost-reduction initiative"],
    "prospect": ["a potential client", "an investor"],
    "terms":    ["the contract renewal", "the pricing agreement"],
    "party":    ["two team members", "departments"],
    "manager":  ["the VP of Engineering", "the project sponsor"],
    "organization": ["a nonprofit", "a university"],
    "deadline": ["the Q4 deliverables", "the project milestone"],
    "achievement": ["the product launch", "the certification"],
    "recipients":["team members", "customers", "partners"],
    "policy_change": ["the new remote work policy", "updated security requirements"],
    "announcement": ["a new partnership", "the product launch"],
    "cause":    ["environmental cleanup", "tech education for youth"],
    "bug":      ["the login timeout issue", "the data export failure"],

    "project_type": ["Python library", "React application", "microservice", "CLI tool"],
    "endpoint": ["/api/users", "/api/orders", "/api/auth/login"],
    "software": ["Docker", "Kubernetes", "the application"],
    "version":  ["2.0.0", "3.1.0", "1.5.0"],
    "tool":     ["Webpack", "Terraform", "Nginx"],
    "language": ["Python", "TypeScript", "Go"],
    "old_version": ["v1", "Python 2", "React 16"],
    "new_version": ["v2", "Python 3", "React 18"],
    "operation":["database failover", "certificate rotation", "scaling up"],
    "database": ["the user database", "the analytics DB"],
    "standard": ["GDPR", "SOC 2", "HIPAA"],
    "infrastructure": ["the cloud infrastructure", "the on-prem setup"],
    "project":  ["the main repo", "the API project", "the frontend app"],
    "service":  ["the payment service", "the notification system"],
}


# multi-task templeates with placeholders for two tasks

COMPATIBLE_PAIRS: List[Tuple[str, str]] = [
    ("summarization", "translation"),
    ("code", "documentation"),
    ("code", "explanation"),
    ("math", "explanation"),
    ("data_analysis", "summarization"),
    ("translation", "explanation"),
    ("summarization", "communication"),
    ("data_analysis", "communication"),
    ("research", "summarization"),
    ("prediction", "explanation"),
    ("simulation", "explanation"),
    ("documentation", "communication"),
    ("translation", "summarization"),
    ("code", "code"),
    ("math", "math"),
    ("explanation", "explanation"),
    ("translation", "translation"),
]

DEPENDENT_PAIRS: List[Tuple[str, str]] = [
    ("data_analysis", "prediction"),
    ("data_analysis", "summarization"),
    ("research", "prediction"),
    ("simulation", "data_analysis"),
    ("code", "documentation"),
    ("data_analysis", "communication"),
    ("research", "communication"),
    ("math", "code"),
    ("simulation", "explanation"),
    ("prediction", "communication"),
]

INDEPENDENT_CONNECTORS = [
    "{t1}. Also, {t2}.",
    "{t1}. Additionally, {t2}.",
    "{t1}. Separately, {t2}.",
    "{t1}. In addition, {t2}.",
    "{t1}, and also {t2}.",
    "Do two things: first, {t1}. Second, {t2}.",
    "I need help with two tasks: {t1}, and {t2}.",
    "{t1}. While you're at it, {t2}.",
]

DEPENDENT_CONNECTORS = [
    "First, {t1}. Then, based on the result, {t2}.",
    "{t1}. Once that is done, {t2}.",
    "{t1}. Using the output, {t2}.",
    "Start by {t1_gerund}. Then {t2}.",
    "{t1}. After completing that, {t2}.",
    "{t1}. Next, using those results, {t2}.",
    "{t1}. Based on your findings, {t2}.",
    "First {t1}, and then use that to {t2_bare}.",
]

HARD_EXAMPLES = [
    "can u check this and maybe explain what's happening",
    "idk what's going on here can u help",
    "make this shorter and explain it",
    "what does this code even do",
    "convert this but also explain it"
]


# complexity estimation based on intent and multiple signals in the text

def estimate_complexity(text: str, intent: str) -> float:
    """
    Estimate task complexity on [0, 1] using multiple signals.
    Returns a float rounded to 2 decimals.
    """
    t = text.lower()
    words = t.split()
    wc = len(words)

    intent_priors = {
        "math": 0.60, "code": 0.65, "simulation": 0.70,
        "research": 0.55, "prediction": 0.55, "data_analysis": 0.60,
        "translation": 0.25, "summarization": 0.25,
        "explanation": 0.30, "communication": 0.20, "documentation": 0.25,
    }
    base = intent_priors.get(intent, 0.35)

    length_bonus = min(wc / 50.0, 0.15)

    math_kw = ["integral", "derivative", "eigenvalue", "matrix", "prove",
               "differential", "fourier", "laplace", "convergence", "optimization"]
    math_signal = sum(1 for kw in math_kw if kw in t) * 0.04

    code_kw = ["implement", "algorithm", "data structure", "concurrent",
               "thread-safe", "optimize", "binary tree", "graph", "dynamic programming"]
    code_signal = sum(1 for kw in code_kw if kw in t) * 0.04

    reason_kw = ["analyze", "compare", "evaluate", "synthesize", "critique",
                 "trade-off", "implications", "derive", "prove"]
    reason_signal = sum(1 for kw in reason_kw if kw in t) * 0.03

    multi_kw = ["step by step", "first", "then", "finally", "multiple"]
    multi_signal = sum(1 for kw in multi_kw if kw in t) * 0.03

    score = base + length_bonus + math_signal + code_signal + reason_signal + multi_signal
    return round(min(max(score, 0.10), 0.95), 2)

def generate_confusing_prompt(rng):
    templates = [
        "can u explain this and make it short",
        "convert this to french and explain it",
        "what's going on here and can u summarize",
        "analyze this and also write code for it",
        "translate this and tell me what it means",
        "summarize this but also explain in detail",
        "help me understand this and code it",
    ]
    return rng.choice(templates)


def fill_template(template: str, rng: random.Random) -> str:
    """Replace all {placeholder} tokens with random values from FILL_BANK."""
    def replacer(match):
        key = match.group(1)
        if key in FILL_BANK:
            return rng.choice(FILL_BANK[key])
        return match.group(0)  # leave unknown placeholders

    filled = re.sub(r"\{(\w+)\}", replacer, template)
    # Clean up any double spaces
    filled = re.sub(r"\s+", " ", filled).strip()
    if rng.random() < 0.2:
        filled = filled.replace("Calculate", "Can you calculate")
    if rng.random() < 0.1:
        filled = rng.choice(HARD_EXAMPLES)

    return filled



INTENT_DETECT_RULES: Dict[str, List[str]] = {
    "code":          ["write code", "implement", "function", "algorithm", "class", "program",
                      "debug", "unit test", "API endpoint", "script", "regex", "sort",
                      "data structure", "linked list", "binary", "hash map", "trie"],
    "math":          ["solve", "calculate", "compute", "integral", "derivative", "prove",
                      "eigenvalue", "matrix", "equation", "probability", "limit", "sum of",
                      "determinant", "polynomial", "GCD", "permutation", "factorial",
                      "standard deviation", "variance", "z-score"],
    "simulation":    ["simulate", "simulation", "monte carlo", "model the", "N-body",
                      "random walk", "agent-based", "queuing", "trajectory", "cellular automaton"],
    "research":      ["survey", "literature", "state-of-the-art", "research gap",
                      "compare and contrast", "critically evaluate", "meta-analysis",
                      "systematic review", "open research questions"],
    "prediction":    ["predict", "forecast", "estimate future", "projection", "trend",
                      "demand forecast", "likelihood", "churn rate", "ROI", "market size"],
    "data_analysis": ["analyze", "correlation", "outlier", "cluster", "regression",
                      "hypothesis test", "sentiment", "pivot table", "anomaly",
                      "principal component", "confusion matrix", "ROC curve", "ANOVA",
                      "feature importance", "descriptive statistics"],
    "translation":   ["translate", "translation", "how do you say", "convert to",
                      "in french", "in spanish", "in german", "in japanese", "in mandarin"],
    "summarization": ["summarize", "summary", "TL;DR", "executive summary", "condense",
                      "key takeaways", "brief overview", "recap", "abstract"],
    "explanation":   ["explain", "what is", "how does", "difference between", "describe",
                      "why does", "what are", "fundamentals"],
    "communication": ["draft", "email", "letter", "compose", "message", "follow-up",
                      "announcement", "cover letter", "proposal email", "invitation"],
    "documentation": ["README", "API documentation", "user guide", "changelog",
                      "troubleshooting guide", "runbook", "onboarding guide",
                      "migration guide", "style guide", "architecture decision"],
}



def detect_intent(text: str) -> str:
    """Rule-based intent detection aligned with canonical labels."""
    t = text.lower()
    scores: Dict[str, int] = {label: 0 for label in INTENT_LABELS}

    for intent, keywords in INTENT_DETECT_RULES.items():
        for kw in keywords:
            if kw.lower() in t:
                scores[intent] += 1

    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return "explanation"
    return best



def is_quality_prompt(text: str) -> bool:
    """Check if a generated prompt meets quality standards."""
    if not text or not text.strip():
        return False
    words = text.split()
    if len(words) < 3:
        return False
    if len(words) > 80:
        return False
    if text.count("{") > 0:  # unfilled template
        return False
    return True


def text_fingerprint(text: str) -> str:
    """Create a fingerprint for near-duplicate detection."""
    # Normalize: lowercase, remove punctuation, sort words
    t = re.sub(r"[^\w\s]", "", text.lower())
    tokens = sorted(t.split())
    trigrams = [" ".join(tokens[i:i+4]) for i in range(len(tokens)-2)]
    combined = "|".join(trigrams[:10])  # first 10 trigrams
    return hashlib.md5(combined.encode()).hexdigest()



def paraphrase_with_llm(text: str, model_name: str = "gemini-pro") -> Optional[str]:
    """Paraphrase into messy, realistic human queries."""
    if not GENAI_AVAILABLE:
        return None

    try:
        model = genai.GenerativeModel(model_name)

        prompt = f"""
Rewrite this like a REAL user query.

Rules:
- Make it casual (like chat / whatsapp)
- Can mix multiple tasks
- Can include multiple intents
- Avoid formal words like calculate, analyze
- Can be slightly messy

Example:
"calculate sum" → "yo can u figure this out"

Original:
{text}

Output only rewritten query.
"""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=100,
                temperature=0.9,  
            ),
        )

        result = getattr(response, "text", "").strip()

        # relaxed filter
        if result and len(result.split()) >= 3:
            return result

    except Exception as e:
        log.debug(f"LLM paraphrase failed: {e}")

    return None

def assign_route_realistic(intent, complexity, rng):

    # base decision
    if intent in STRONG_INTENTS:
        route = "strong_model" if complexity >= 0.4 else "weak_model"
    elif intent in WEAK_INTENTS:
        route = "strong_model" if complexity >= 0.6 else "weak_model"
    else:
        route = "strong_model" if complexity > 0.5 else "weak_model"

    # 🔥 ADD NOISE (IMPORTANT)
    if rng.random() < 0.15:   # 15% noise
        route = "weak_model" if route == "strong_model" else "strong_model"

    return route


def generate_single_task(
    intent: str,
    rng: random.Random,
    prompt_id: int,
    use_llm: bool = False,
) -> Optional[Dict[str, Any]]:
    """Generate one single-task sample in the canonical JSONL format."""
    templates = TEMPLATES.get(intent, [])
    if not templates:
        return None

    template = rng.choice(templates)
    if rng.random() < 0.2:
        text = generate_confusing_prompt(rng)
    else:
        text = fill_template(template, rng)
    text = make_natural(text,rng)
    text = add_noise(text,rng)


    if not is_quality_prompt(text):
        return None

    # Optional LLM paraphrase
    if use_llm and rng.random() < 0.8:
        paraphrased = paraphrase_with_llm(text)
        if paraphrased:
            text = paraphrased

    detected = detect_intent(text)

    complexity = estimate_complexity(text, intent)
    
    model_req = assign_route_realistic(intent, complexity, rng)
    # HARD ROUTING INJECTION
    if rng.random() < 0.1:
        text, intent, complexity, model_req = rng.choice(HARD_ROUTING)

    return {
        "prompt_id": prompt_id,
        "prompt": text,
        "segments": [{
            "segment_id": 1,
            "text": text,
            "intent": intent,
            "complexity_score": complexity,
            "model_requirement": model_req,
            "depends_on": [],
        }],
    }


def generate_multi_task(
    pair: Tuple[str, str],
    is_dependent: bool,
    rng: random.Random,
    prompt_id: int,
    use_llm: bool = False,
) -> Optional[Dict[str, Any]]:
    """Generate one multi-task sample with 2 segments."""
    intent1, intent2 = pair

    t1_templates = TEMPLATES.get(intent1, [])
    t2_templates = TEMPLATES.get(intent2, [])
    if not t1_templates or not t2_templates:
        return None


    if rng.random() < 0.2:
        t1_text = generate_confusing_prompt(rng)
    else:
        t1_text = fill_template(rng.choice(t1_templates), rng)
    if rng.random() < 0.2:
        t2_text = generate_confusing_prompt(rng)
    else:
        t2_text = fill_template(rng.choice(t2_templates), rng)

    t1_text=make_natural(t1_text,rng)
    t2_text=make_natural(t2_text,rng)
    t1_text=add_noise(t1_text,rng)
    t2_text=add_noise(t2_text,rng)


    if not is_quality_prompt(t1_text) or not is_quality_prompt(t2_text):
        return None

    if is_dependent:
        connector = rng.choice(DEPENDENT_CONNECTORS)
        t1_gerund = t1_text[0].lower() + t1_text[1:] if t1_text else t1_text
        t2_bare = t2_text[0].lower() + t2_text[1:] if t2_text else t2_text
        prompt = connector.format(
            t1=t1_text, t2=t2_text,
            t1_gerund=t1_gerund, t2_bare=t2_bare,
        )
    else:
        connector = rng.choice(INDEPENDENT_CONNECTORS)
        prompt = connector.format(t1=t1_text, t2=t2_text)

    if not is_quality_prompt(prompt):
        return None

    c1 = estimate_complexity(t1_text, intent1)
    c2 = estimate_complexity(t2_text, intent2)
    m1 = assign_route_realistic(intent1, c1, rng)
    m2 = assign_route_realistic(intent2, c2, rng)

    depends_on = [1] if is_dependent else []
    # HARD ROUTING INJECTION
    if rng.random() < 0.1:
        t1_text, intent1, c1, m1 = rng.choice(HARD_ROUTING)
    if rng.random() < 0.1:
        t2_text, intent2, c2, m2 = rng.choice(HARD_ROUTING)

    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "segments": [
            {
                "segment_id": 1,
                "text": t1_text,
                "intent": intent1,
                "complexity_score": c1,
                "model_requirement": m1,
                "depends_on": [],
            },
            {
                "segment_id": 2,
                "text": t2_text,
                "intent": intent2,
                "complexity_score": c2,
                "model_requirement": m2,
                "depends_on": depends_on,
            },
        ],
    }


def generate_dataset(
    num_samples: int = 6000,
    multi_task_ratio: float = 0.90,
    dependent_ratio: float = 0.40,
    seed: int = 42,
    use_llm: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate the full dataset with balanced intents.

    Returns:
        (dataset, statistics)
    """
    rng = random.Random(seed)
    dataset: List[Dict[str, Any]] = []
    seen_fingerprints: set = set()
    prompt_id = 1

    num_multi = int(num_samples * multi_task_ratio)
    num_single = num_samples - num_multi

    # ── Balanced single-task generation ─────────────────────
    per_intent = num_single // len(INTENT_LABELS)
    remainder = num_single % len(INTENT_LABELS)

    intent_targets = {label: per_intent for label in INTENT_LABELS}
    # Distribute remainder
    for i, label in enumerate(INTENT_LABELS):
        if i < remainder:
            intent_targets[label] += 1

    log.info(f"Target: {num_single} single-task, {num_multi} multi-task")
    log.info(f"Per-intent target: ~{per_intent} samples each")

    intent_counts: Dict[str, int] = {label: 0 for label in INTENT_LABELS}
    max_attempts_per_intent = per_intent * 25

    for intent in INTENT_LABELS:
        target = intent_targets[intent]
        attempts = 0
        while intent_counts[intent] < target and attempts < max_attempts_per_intent:
            attempts += 1
            sample = generate_single_task(intent, rng, prompt_id, use_llm)
            if sample is None:
                continue

            fp = text_fingerprint(sample["prompt"])
            if fp in seen_fingerprints:
                continue
            seen_fingerprints.add(fp)

            dataset.append(sample)
            intent_counts[intent] += 1
            prompt_id += 1

        if intent_counts[intent] < target:
            log.warning(f"  {intent}: generated {intent_counts[intent]}/{target}")

        while intent_counts[intent] < target:
            sample = generate_single_task(intent, rng, prompt_id, use_llm=False)

            if sample:
                dataset.append(sample)
                intent_counts[intent] += 1
                prompt_id += 1

    log.info(f"Single-task generated: {len(dataset)}")

    num_dependent = int(num_multi * dependent_ratio)
    num_independent = num_multi - num_dependent
    multi_count = 0

    for _ in range(num_independent * 5):
        if multi_count >= num_independent:
            break
        pair = rng.choice(COMPATIBLE_PAIRS)
        sample = generate_multi_task(pair, False, rng, prompt_id, use_llm)
        if sample is None:
            continue
        fp = text_fingerprint(sample["prompt"])
        if fp in seen_fingerprints:
            continue
        seen_fingerprints.add(fp)
        dataset.append(sample)
        multi_count += 1
        prompt_id += 1

    log.info(f"Independent multi-task generated: {multi_count}")

    dep_count = 0
    for _ in range(num_dependent * 5):
        if dep_count >= num_dependent:
            break
        pair = rng.choice(DEPENDENT_PAIRS)
        sample = generate_multi_task(pair, True, rng, prompt_id, use_llm)
        if sample is None:
            continue
        fp = text_fingerprint(sample["prompt"])
        if fp in seen_fingerprints:
            continue
        seen_fingerprints.add(fp)
        dataset.append(sample)
        dep_count += 1
        prompt_id += 1

    log.info(f"Dependent multi-task generated: {dep_count}")

    rng.shuffle(dataset)

    for i, sample in enumerate(dataset):
        sample["prompt_id"] = i + 1

    stats = compute_statistics(dataset)

    return dataset, stats


def compute_statistics(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute dataset statistics for the data card."""
    total = len(dataset)
    single = sum(1 for d in dataset if len(d["segments"]) == 1)
    multi = total - single
    dependent = sum(
        1 for d in dataset
        if any(seg["depends_on"] for seg in d["segments"])
    )

    intent_dist: Counter = Counter()
    complexity_scores: List[float] = []
    strong_count = 0
    weak_count = 0
    word_counts: List[int] = []

    for d in dataset:
        word_counts.append(len(d["prompt"].split()))
        for seg in d["segments"]:
            intent_dist[seg["intent"]] += 1
            complexity_scores.append(seg["complexity_score"])
            if seg["model_requirement"] == ROUTE_STRONG:
                strong_count += 1
            else:
                weak_count += 1

    total_segments = sum(intent_dist.values())

    return {
        "total_prompts": total,
        "single_task_prompts": single,
        "multi_task_prompts": multi,
        "multi_task_ratio": round(multi / total, 3) if total else 0,
        "dependent_prompts": dependent,
        "dependent_ratio": round(dependent / multi, 3) if multi else 0,
        "total_segments": total_segments,
        "intent_distribution": {
            k: {"count": v, "pct": round(v / total_segments * 100, 1)}
            for k, v in sorted(intent_dist.items())
        },
        "complexity": {
            "mean": round(sum(complexity_scores) / len(complexity_scores), 3) if complexity_scores else 0,
            "min": min(complexity_scores) if complexity_scores else 0,
            "max": max(complexity_scores) if complexity_scores else 0,
        },
        "routing": {
            "strong_model_segments": strong_count,
            "weak_model_segments": weak_count,
            "strong_pct": round(strong_count / total_segments * 100, 1) if total_segments else 0,
        },
        "prompt_length": {
            "mean_words": round(sum(word_counts) / len(word_counts), 1) if word_counts else 0,
            "min_words": min(word_counts) if word_counts else 0,
            "max_words": max(word_counts) if word_counts else 0,
        },
    }

def add_noise(text, rng):
    if rng.random() < 0.2:
        text = text.replace("you", "u")
    if rng.random() < 0.2:
        text = text.lower()
    if rng.random() < 0.1:
        text = text.replace("the", "")
    return text


def save_jsonl(dataset: List[Dict[str, Any]], path: Path) -> None:
    """Save dataset in JSONL format (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info(f"Saved {len(dataset)} samples to {path}")


def save_statistics(stats: Dict[str, Any], path: Path) -> None:
    """Save dataset statistics as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log.info(f"Saved statistics to {path}")



def main():
    parser = argparse.ArgumentParser(
        description="Generate research-grade synthetic dataset for Adaptive Semantic Parallelism"
    )
    parser.add_argument(
        "--num-samples", type=int, default=6000,
        help="Total number of prompts to generate (default: 6000)"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/dataset.jsonl"),
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--stats-output", type=Path,
        default=None,
        help="Output path for dataset statistics JSON (default: alongside dataset)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--multi-task-ratio", type=float, default=0.30,
        help="Fraction of multi-task prompts (default: 0.30)"
    )
    parser.add_argument(
        "--dependent-ratio", type=float, default=0.40,
        help="Fraction of multi-task prompts that are dependent (default: 0.40)"
    )
    parser.add_argument(
        "--use-llm", action="store_true",
        help="Use Gemini API to paraphrase some prompts (requires GEMINI_API_KEY)"
    )

    args = parser.parse_args()

    if args.use_llm:
        if not GENAI_AVAILABLE:
            log.error("google-generativeai not installed. pip install google-generativeai")
            return
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            log.error("GEMINI_API_KEY environment variable not set")
            return
        genai.configure(api_key=api_key)
        log.info("LLM paraphrasing enabled (Gemini)")

    dataset, stats = generate_dataset(
        num_samples=args.num_samples,
        multi_task_ratio=args.multi_task_ratio,
        dependent_ratio=args.dependent_ratio,
        seed=args.seed,
        use_llm=args.use_llm,
    )

    save_jsonl(dataset, args.output)

    stats_path = args.stats_output or args.output.with_name("dataset_stats.json")
    save_statistics(stats, stats_path)

    # Print summary
    print("\n" + "=" * 60)
    print("  DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Total prompts:      {stats['total_prompts']}")
    print(f"  Total segments:     {stats['total_segments']}")
    print(f"  Single-task:        {stats['single_task_prompts']}")
    print(f"  Multi-task:         {stats['multi_task_prompts']} ({stats['multi_task_ratio']*100:.1f}%)")
    print(f"  Dependent:          {stats['dependent_prompts']} ({stats['dependent_ratio']*100:.1f}% of multi)")
    print(f"  Strong-routed:      {stats['routing']['strong_model_segments']} ({stats['routing']['strong_pct']}%)")
    print(f"  Weak-routed:        {stats['routing']['weak_model_segments']}")
    print(f"  Avg complexity:     {stats['complexity']['mean']}")
    print(f"  Avg prompt length:  {stats['prompt_length']['mean_words']} words")
    print()
    print("  Intent distribution:")
    for intent, info in stats["intent_distribution"].items():
        bar = "█" * int(info["pct"] / 2)
        print(f"    {intent:20s}  {info['count']:5d}  ({info['pct']:5.1f}%)  {bar}")
    print("=" * 60)


if __name__ == "__main__":
    main()