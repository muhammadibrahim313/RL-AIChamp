"""
Outlier Detection with Treatment Justification - RL Task for LLM Training
Target success rate: 30-35%
An ML engineer must detect outliers, decide on appropriate treatment, and justify decisions.
"""

import asyncio
import json
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Callable, TypedDict
import numpy as np
import pandas as pd

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam


class PythonExpressionToolResult(TypedDict):
    result: Any
    error: str | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def python_expression_tool(expression: str) -> PythonExpressionToolResult:
    """
    Tool that evaluates Python expressions using exec.
    Use print(...) to emit output; stdout will be captured and returned.
    Pre-imports: numpy as np, pandas as pd
    """
    try:
        namespace = {"np": np, "pd": pd}
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(expression, namespace, namespace)
        return {"result": stdout.getvalue(), "error": None}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"result": None, "error": str(e)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """
    Tool for submitting the final answer.
    Answer should be a JSON string with:
    {
        "outliers_detected": [...],  # List of column names with outliers
        "treatment_method": {...},   # Dict mapping column -> treatment (remove/cap/transform)
        "justification": {...}       # Dict mapping column -> reasoning
    }
    """
    return {"answer": answer, "submitted": True}


def generate_dataset_with_outliers() -> tuple[pd.DataFrame, dict]:
    """
    Generate a realistic dataset with intentional outliers and ground truth.
    Returns: (dataframe, ground_truth)
    """
    np.random.seed(42)
    n_samples = 500
    
    # Create base dataset
    data = {
        # Normal distribution - should use IQR or Z-score
        'age': np.random.normal(35, 10, n_samples),
        
        # Right-skewed with legitimate high values - should cap, not remove
        'income': np.random.lognormal(10.5, 0.8, n_samples),
        
        # Contains measurement errors - should remove
        'heart_rate': np.random.normal(75, 10, n_samples),
        
        # Normal with extreme valid cases - should transform (log)
        'transaction_amount': np.random.gamma(2, 50, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Add realistic outliers
    # Age: Some data entry errors (negative, >150)
    df.loc[5, 'age'] = -5
    df.loc[23, 'age'] = 200
    df.loc[47, 'age'] = -1
    
    # Income: Some extreme but valid high earners (don't remove, cap)
    df.loc[10, 'income'] = 500000
    df.loc[89, 'income'] = 750000
    
    # Heart rate: Clear measurement errors (should remove)
    df.loc[15, 'heart_rate'] = 0
    df.loc[78, 'heart_rate'] = 250
    df.loc[123, 'heart_rate'] = -10
    
    # Transaction: Few very large transactions (valid, maybe log transform)
    df.loc[34, 'transaction_amount'] = 5000
    df.loc[156, 'transaction_amount'] = 8000
    
    # Ground truth for grading
    ground_truth = {
        'outliers_detected': ['age', 'heart_rate', 'income', 'transaction_amount'],
        'treatment_method': {
            'age': 'remove',  # Clear errors
            'heart_rate': 'remove',  # Measurement errors
            'income': 'cap',  # Valid extremes
            'transaction_amount': 'transform'  # Right-skewed
        },
        'justification_keywords': {
            'age': ['negative', 'impossible', 'error', 'invalid'],
            'heart_rate': ['measurement', 'error', 'impossible', 'invalid'],
            'income': ['valid', 'extreme', 'cap', 'percentile', 'legitimate'],
            'transaction_amount': ['skewed', 'log', 'transform', 'distribution']
        }
    }
    
    return df, ground_truth


def grade_answer(submitted_answer: Any, ground_truth: dict) -> tuple[int, str]:
    """
    Grade the submitted answer on a 0-3 scale (low-precision for consistency).
    0: No outlier detection or completely wrong
    1: Detected some outliers but poor treatment decisions
    2: Good detection and treatment, weak justification
    3: Excellent detection, treatment, and justification
    """
    try:
        # Parse answer
        if isinstance(submitted_answer, str):
            answer = json.loads(submitted_answer)
        elif isinstance(submitted_answer, dict):
            answer = submitted_answer
        else:
            return 0, "Invalid format: answer must be JSON string or dict"
        
        # Check required keys
        required_keys = ['outliers_detected', 'treatment_method', 'justification']
        if not all(key in answer for key in required_keys):
            return 0, f"Missing required keys. Need: {required_keys}"
        
        outliers_detected = set(answer['outliers_detected'])
        treatment_method = answer['treatment_method']
        justification = answer['justification']
        
        # Score detection (max 1 point)
        expected_outliers = set(ground_truth['outliers_detected'])
        detected_correctly = len(outliers_detected & expected_outliers)
        detection_score = detected_correctly / len(expected_outliers)
        
        if detection_score < 0.5:
            return 0, f"Poor detection: found {detected_correctly}/{len(expected_outliers)} columns"
        
        # Score treatment decisions (max 1 point)
        correct_treatments = 0
        total_treatments = 0
        treatment_details = []
        
        for col in expected_outliers:
            if col in treatment_method:
                total_treatments += 1
                expected = ground_truth['treatment_method'][col]
                actual = treatment_method[col].lower()
                
                # Flexible matching
                if expected == 'remove' and any(word in actual for word in ['remove', 'drop', 'delete']):
                    correct_treatments += 1
                    treatment_details.append(f"✓ {col}: correct (remove)")
                elif expected == 'cap' and any(word in actual for word in ['cap', 'clip', 'winsorize', 'percentile']):
                    correct_treatments += 1
                    treatment_details.append(f"✓ {col}: correct (cap)")
                elif expected == 'transform' and any(word in actual for word in ['transform', 'log', 'sqrt', 'box-cox']):
                    correct_treatments += 1
                    treatment_details.append(f"✓ {col}: correct (transform)")
                else:
                    treatment_details.append(f"✗ {col}: expected {expected}, got {actual}")
        
        treatment_score = correct_treatments / len(expected_outliers) if expected_outliers else 0
        
        # Score justification quality (max 1 point)
        justification_score = 0
        justification_details = []
        
        for col in expected_outliers:
            if col in justification and col in ground_truth['justification_keywords']:
                justif_text = str(justification[col]).lower()
                keywords = ground_truth['justification_keywords'][col]
                
                # Check if justification contains relevant keywords
                keyword_matches = sum(1 for kw in keywords if kw in justif_text)
                if keyword_matches >= 1:
                    justification_score += 1
                    justification_details.append(f"✓ {col}: good reasoning")
                else:
                    justification_details.append(f"✗ {col}: weak justification")
        
        justification_score /= len(expected_outliers) if expected_outliers else 1
        
        # Calculate final score (0-3 scale)
        total_score = detection_score + treatment_score + justification_score
        
        if total_score < 1.0:
            final_score = 0
            explanation = "Failed: Poor outlier detection"
        elif total_score < 2.0:
            final_score = 1
            explanation = "Partial: Detected outliers but treatment/justification weak"
        elif total_score < 2.5:
            final_score = 2
            explanation = "Good: Solid detection and treatment, justification needs work"
        else:
            final_score = 3
            explanation = "Excellent: Strong detection, treatment, and justification"
        
        details = f"""
{explanation}
Detection: {detected_correctly}/{len(expected_outliers)} columns
Treatment: {treatment_details}
Justification: {justification_details}
Scores: detection={detection_score:.2f}, treatment={treatment_score:.2f}, justification={justification_score:.2f}
"""
        return final_score, details.strip()
        
    except json.JSONDecodeError as e:
        return 0, f"JSON parsing error: {e}"
    except Exception as e:
        return 0, f"Grading error: {e}"


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 10,
    model: str = "claude-sonnet-4-20250514",
    verbose: bool = True,
) -> Any | None:
    """
    Runs an agent loop with the given prompt and tools.
    """
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        response = await client.messages.create(
            model=model, max_tokens=2000, tools=tools, messages=messages
        )

        has_tool_use = False
        tool_results = []
        submitted_answer = None

        for content in response.content:
            if content.type == "text":
                if verbose:
                    print(f"Assistant: {content.text}")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name

                if tool_name in tool_handlers:
                    if verbose:
                        print(f"Using tool: {tool_name}")

                    handler = tool_handlers[tool_name]
                    tool_input = content.input

                    if tool_name == "python_expression":
                        assert (
                            isinstance(tool_input, dict) and "expression" in tool_input
                        )
                        if verbose:
                            print("\nInput:")
                            print("```python")
                            print(tool_input["expression"])
                            print("```")
                        result = handler(tool_input["expression"])
                        if verbose:
                            print("\nOutput:")
                            print(result)
                    elif tool_name == "submit_answer":
                        assert isinstance(tool_input, dict) and "answer" in tool_input
                        result = handler(tool_input["answer"])
                        submitted_answer = result["answer"]
                    else:
                        result = (
                            handler(**tool_input)
                            if isinstance(tool_input, dict)
                            else handler(tool_input)
                        )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": json.dumps(result),
                        }
                    )

        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            if submitted_answer is not None:
                if verbose:
                    print(f"\nAgent submitted answer")
                return submitted_answer
        else:
            if verbose:
                print("\nNo tool use in response, ending loop.")
            break

    if verbose:
        print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
    return None


async def run_single_test(
    run_id: int,
    num_runs: int,
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    ground_truth: dict,
    verbose: bool = False,
) -> tuple[int, int, Any]:
    """Run a single test and return (run_id, score, result)"""
    if verbose:
        print(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} {'=' * 20}")

    result = await run_agent_loop(
        prompt=prompt,
        tools=tools,
        tool_handlers=tool_handlers,
        max_steps=10,
        model="claude-sonnet-4-20250514",
        verbose=verbose,
    )

    score, explanation = grade_answer(result, ground_truth)

    if score >= 2:
        print(f"✓ Run {run_id}: SUCCESS (score={score}/3)")
    else:
        print(f"✗ Run {run_id}: FAILURE (score={score}/3)")
    
    if verbose:
        print(f"\n{explanation}")

    return run_id, score, result


async def main(concurrent: bool = True):
    # Generate dataset and ground truth
    df, ground_truth = generate_dataset_with_outliers()
    
    # Save dataset to CSV for the task (in current directory)
    df.to_csv('outlier_dataset.csv', index=False)
    
    # Create the task prompt
    prompt = """You are a data scientist working on a machine learning project. You've been given a dataset that may contain outliers.

Your task:
1. Load the dataset from 'outlier_dataset.csv'
2. Detect outliers in numerical columns using BOTH IQR method AND Z-score
3. For EACH column with outliers, decide on the appropriate treatment:
   - REMOVE: if outliers are clearly measurement errors or impossible values
   - CAP: if outliers are extreme but potentially valid values
   - TRANSFORM: if the distribution is skewed and transformation would help
4. Provide clear justification for each decision based on:
   - The nature of the data (what does it represent?)
   - Distribution characteristics (skewness, range, validity)
   - Whether outliers seem like errors vs. legitimate extreme values

Use the python_expression tool to:
- Load and explore the data
- Calculate outlier thresholds
- Visualize distributions if helpful
- Analyze the characteristics of outliers

Then submit your answer as a JSON string with this structure:
{
    "outliers_detected": ["column1", "column2", ...],
    "treatment_method": {
        "column1": "remove",  # or "cap" or "transform"
        "column2": "cap",
        ...
    },
    "justification": {
        "column1": "Explanation of why you chose this treatment",
        "column2": "Explanation...",
        ...
    }
}

IMPORTANT: 
- Think carefully about whether outliers are errors or valid extremes
- Your justifications should reference specific characteristics of the data
- Consider the real-world meaning of each variable
"""

    tools: list[ToolUnionParam] = [
        {
            "name": "python_expression",
            "description": "Evaluates Python code. numpy (as np) and pandas (as pd) are pre-imported.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Python code to execute. Use print() to see output.",
                    }
                },
                "required": ["expression"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit your final answer as a JSON string",
            "input_schema": {
                "type": "object",
                "properties": {
                    "answer": {
                        "description": "JSON string with outliers_detected, treatment_method, and justification"
                    }
                },
                "required": ["answer"],
            },
        },
    ]

    tool_handlers = {
        "python_expression": python_expression_tool,
        "submit_answer": submit_answer_tool,
    }

    # Run the test 10 times
    num_runs = 10
    execution_mode = "concurrently" if concurrent else "sequentially"
    print(f"Running {num_runs} test iterations {execution_mode}...")
    print(f"Target success rate: 30-35% (scores of 2-3 out of 3)")
    print("=" * 60)

    tasks = [
        run_single_test(
            run_id=i + 1,
            num_runs=num_runs,
            prompt=prompt,
            tools=tools,
            tool_handlers=tool_handlers,
            ground_truth=ground_truth,
            verbose=False,
        )
        for i in range(num_runs)
    ]

    if concurrent:
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
    else:
        results = []
        for task in tasks:
            result = await task
            results.append(result)

    # Calculate statistics
    scores = [score for _, score, _ in results]
    successes = sum(1 for score in scores if score >= 2)  # 2-3 are successes
    excellent = sum(1 for score in scores if score == 3)
    good = sum(1 for score in scores if score == 2)
    partial = sum(1 for score in scores if score == 1)
    failed = sum(1 for score in scores if score == 0)
    
    success_rate = (successes / num_runs) * 100
    avg_score = sum(scores) / len(scores)

    print(f"\n{'=' * 60}")
    print("Test Results:")
    print(f"  Excellent (3/3): {excellent}/{num_runs}")
    print(f"  Good (2/3):      {good}/{num_runs}")
    print(f"  Partial (1/3):   {partial}/{num_runs}")
    print(f"  Failed (0/3):    {failed}/{num_runs}")
    print(f"  Success Rate:    {success_rate:.1f}% (target: 30-35%)")
    print(f"  Average Score:   {avg_score:.2f}/3.0")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Run sequentially to avoid rate limits with paid tier
    asyncio.run(main(concurrent=False))
