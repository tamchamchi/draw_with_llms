import ast
import os
import statistics
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd

from src.data.image_processor import ImageProcessor, svg_to_png
from src.data import svg_constraints

from src.models.global_models import vqa_evaluator, aesthetic_evaluator


class ParticipantVisibleError(Exception):
    pass


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    random_seed: int = 0,
) -> float:
    """Calculates a fidelity score by comparing generated SVG images to target text descriptions.

    Parameters
    ----------
    solution : pd.DataFrame
         A DataFrame containing target questions, choices, and answers about an SVG image.
    submission : pd.DataFrame
         A DataFrame containing generated SVG strings. Must have a column named 'svg'.
    row_id_column_name : str
         The name of the column containing row identifiers. This column is removed before scoring.
    random_seed : int
         A seed to set the random state.

    Returns
    -------
    float
         The mean fidelity score (a value between 0 and 1) representing the average similarity between the generated SVGs and their descriptions.
         A higher score indicates better fidelity.

    Raises
    ------
    ParticipantVisibleError
         If the 'svg' column in the submission DataFrame is not of string type or if validation of the SVG fails.

    Examples
    --------
    >>> import pandas as pd
    >>> solution = pd.DataFrame({
    ...     'id': ["abcde"],
    ...     'question': ['["Is there a red circle?", "What shape is present?"]'],
    ...     'choices': ['[["yes", "no"], ["square", "circle", "triangle", "hexagon"]]'],
    ...     'answer': ['["yes", "circle"]'],
    ... })
    >>> submission = pd.DataFrame({
    ...     'id': ["abcde"],
    ...     'svg': ['<svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="red"/></svg>'],
    ... })
    >>> score(solution, submission, 'row_id', random_seed=42)
    0...
    """
    # Convert solution fields to list dtypes and expand
    for colname in ["question", "choices", "answer"]:
        solution[colname] = solution[colname].apply(ast.literal_eval)
    solution = solution.explode(["question", "choices", "answer"])

    # Validate
    if not pd.api.types.is_string_dtype(submission.loc[:, "svg"]):
        raise ParticipantVisibleError("svg must be a string.")

    # Check that SVG code meets defined constraints
    constraints = svg_constraints.SVGConstraints()
    try:
        for svg in submission.loc[:, "svg"]:
            constraints.validate_svg(svg)
    except:
        raise ParticipantVisibleError("SVG code violates constraints.")

    # Score
    # vqa_evaluator = VQAEvaluator()
    # aesthetic_evaluator = AestheticEvaluator()

    results = []
    rng = np.random.RandomState(random_seed)
    # try:
    df = solution.merge(submission, on="id")
    for i, (_, group) in enumerate(
        df.loc[:, ["id", "question", "choices", "answer", "svg"]].groupby("id")
    ):
        questions, choices, answers, svg = [
            group[col_name].to_list() for col_name in group.drop("id", axis=1).columns
        ]
        svg = svg[0]  # unpack singleton from list
        group_seed = rng.randint(0, np.iinfo(np.int32).max)
        image_processor = ImageProcessor(image=svg_to_png(svg), seed=group_seed).apply()
        image = image_processor.image.copy()
        aesthetic_score = aesthetic_evaluator.score(image)
        vqa_score = vqa_evaluator.score(questions, choices, answers, image)
        image_processor.reset().apply_random_crop_resize().apply_jpeg_compression(
            quality=90
        )
        ocr_score = vqa_evaluator.ocr(image_processor.image)
        instance_score = harmonic_mean(vqa_score, aesthetic_score, beta=0.5) * ocr_score
        results.append(instance_score)

    # except:
    #     raise ParticipantVisibleError('SVG failed to score.')
    # ---------------Modified output of function--------------------#

    fidelity = statistics.mean(results)
    return float(fidelity), vqa_score, aesthetic_score, ocr_score


def harmonic_mean(a: float, b: float, beta: float = 1.0) -> float:
    """
    Calculate the harmonic mean of two values, weighted using a beta parameter.

    Args:
         a: First value (e.g., precision)
         b: Second value (e.g., recall)
         beta: Weighting parameter

    Returns:
         Weighted harmonic mean
    """
    # Handle zero values to prevent division by zero
    if a <= 0 or b <= 0:
        return 0.0
    return (1 + beta**2) * (a * b) / (beta**2 * a + b)
