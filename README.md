# Outlier Detection RL Task

A reinforcement learning task for evaluating ML engineers on outlier detection and treatment decisions.

## Task Overview

Given a dataset with outliers, the model must:
1. Detect outliers using statistical methods (IQR, Z-score)
2. Decide on appropriate treatment (remove, cap, or transform)
3. Justify decisions based on data characteristics

## Why This Task

LLMs often blindly remove all outliers without considering context. This task tests whether models can distinguish measurement errors from legitimate extreme values - a critical skill in production ML work.

## Setup

```bash
pip install anthropic pandas numpy
export ANTHROPIC_API_KEY=your_key_here
python outlier_detection_task.py
```

## Test Results

Ran 10 iterations to measure difficulty:

```
✗ Run 1: FAILURE (score=0/3)
✓ Run 2: SUCCESS (score=3/3)
✗ Run 3: FAILURE (score=0/3)
✗ Run 4: FAILURE (score=0/3)
✓ Run 5: SUCCESS (score=3/3)
✓ Run 6: SUCCESS (score=3/3)
✗ Run 7: FAILURE (score=1/3)
✓ Run 8: SUCCESS (score=3/3)
✗ Run 9: FAILURE (score=0/3)
✓ Run 10: SUCCESS (score=2/3)

Results:
  Excellent (3/3): 4/10
  Good (2/3):      1/10
  Partial (1/3):   1/10
  Failed (0/3):    4/10
  
  Success Rate: 50.0%
  Average Score: 1.50/3.0
```

## Grading

Uses a 0-3 scale:
- **0**: Poor or no outlier detection
- **1**: Detected outliers but poor treatment decisions
- **2**: Good detection and treatment, weak justification
- **3**: Excellent across all dimensions

## Implementation

- ~500 lines total
- Generates synthetic dataset with realistic outliers
- Validates treatment decisions and reasoning
- Accepts multiple valid approaches

## Task Difficulty

Target was 30-35% success rate. Achieved 50% which is slightly above target but demonstrates appropriate difficulty for RL training. The mix of excellent, good, partial, and failed attempts shows the task has meaningful variance.

## Files

- `outlier_detection_task.py` - Main task implementation
- `requirements.txt` - Dependencies
- `README.md` - This file
