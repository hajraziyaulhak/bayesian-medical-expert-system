# 🧠 Bayesian Medical Diagnosis Expert System

A prototype Expert System that uses a **Bayesian Belief Network (BBN)** to perform probabilistic inference for medical diagnosis.

## 📌 Project Overview

Traditional rule-based expert systems provide fixed answers based on predefined rules. This project uses a Bayesian approach to handle **uncertainty** and calculate the probability of a disease based on observed symptoms.

The system takes symptoms such as:

- Fever
- Cough
- Fatigue

as evidence and calculates the posterior probability of **Flu** using Bayesian inference.

## 🎯 Objective

To design and implement an Expert System using a Bayesian Belief Network and demonstrate probabilistic reasoning, uncertainty handling, and decision making.

## 🔗 Bayesian Belief Network

The system uses the following network:

```text
                Flu
              /  |  \
             ↓   ↓   ↓
          Fever Cough Fatigue
