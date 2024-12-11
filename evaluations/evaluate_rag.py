"""
RAG Evaluation Script

This script evaluates the performance of a Retrieval-Augmented Generation (RAG) system
using various metrics from the deepeval library.

Dependencies:
- deepeval
- langchain_openai
- json

Custom modules:
- helper_functions (for RAG-specific operations)
"""

import json
import os
import sys
from typing import List, Tuple

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_openai import ChatOpenAI

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from helper_functions import (
    answer_question_from_context,
    create_question_answer_from_context_chain,
    retrieve_context_per_question,
)


def create_deep_eval_test_cases(
    questions: List[str],
    gt_answers: List[str],
    generated_answers: List[str],
    retrieved_documents: List[str],
) -> List[LLMTestCase]:
    """
    Create a list of LLMTestCase objects for evaluation.

    Args:
        questions (List[str]): List of input questions.
        gt_answers (List[str]): List of ground truth answers.
        generated_answers (List[str]): List of generated answers.
        retrieved_documents (List[str]): List of retrieved documents.

    Returns:
        List[LLMTestCase]: List of LLMTestCase objects.
    """
    return [
        LLMTestCase(
            input=question,
            expected_output=gt_answer,
            actual_output=generated_answer,
            retrieval_context=retrieved_document,
        )
        for question, gt_answer, generated_answer, retrieved_document in zip(
            questions, gt_answers, generated_answers, retrieved_documents
        )
    ]


# Define evaluation metrics
correctness_metric = GEval(
    name="Correctness",
    model="gpt-4o",
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    evaluation_steps=[
        "Determine whether the actual output is factually correct based on the expected output."
    ],
)

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7, model="gpt-4o", include_reason=False
)

answer_relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7, model="gpt-4o", include_reason=True
)


def evaluate_rag(chunks_query_retriever, num_questions: int = 10) -> None:
    """
    Evaluate the RAG system using predefined metrics.

    Args:
        chunks_query_retriever: Function to retrieve context chunks for a given query.
        num_questions (int): Number of questions to evaluate (default: 10).
    """
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=2000)
    question_answer_from_context_chain = create_question_answer_from_context_chain(llm)

    # Load questions and answers from JSON file
    q_a_file_name = "queriesResponses.json"
    try:
        with open(q_a_file_name, "r", encoding="utf-8") as json_file:
            q_a = json.load(json_file)
    except json.JSONDecodeError as e:
        print(f"Error loading JSON file: {e}")
        return

    questions = [qa.get("question", "").strip() for qa in q_a][:num_questions]
    ground_truth_answers = [qa.get("answer", "").strip() for qa in q_a][:num_questions]
    generated_answers = []
    retrieved_documents = []

    # Generate answers and retrieve documents for each question
    for question in questions:
        context = retrieve_context_per_question(question, chunks_query_retriever)
        retrieved_documents.append(context)
        context_string = " ".join(context).strip()
        result = answer_question_from_context(
            question, context_string, question_answer_from_context_chain
        )
        generated_answer = result.get("answer", "").strip()
        if not generated_answer:
            print(f"Empty answer for question: {question}")
            generated_answer = "No answer provided."
        generated_answers.append(generated_answer)

    # Create test cases and validate
    test_cases = create_deep_eval_test_cases(
        questions, ground_truth_answers, generated_answers, retrieved_documents
    )
    for case in test_cases:
        print(case)  # Debug each test case

    # Evaluate with metrics
    try:
        evaluate(
            test_cases=test_cases,
            metrics=[correctness_metric, faithfulness_metric, answer_relevancy_metric],
        )
    except ValueError as e:
        print(f"Evaluation error: {e}")
        for case in test_cases:
            print(case)


if __name__ == "__main__":
    # Add any necessary setup or configuration here
    # Example: evaluate_rag(your_chunks_query_retriever_function)
    pass
