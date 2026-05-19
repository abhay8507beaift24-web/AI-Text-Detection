"""
load_data.py - Dataset loading for AI Text Detection
Supports loading from CSV, HuggingFace datasets, or generating synthetic data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_from_csv(filepath: str) -> pd.DataFrame:
    """Load dataset from a CSV file with 'text' and 'label' columns."""
    df = pd.read_csv(filepath)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must have columns: {required}. Found: {set(df.columns)}")
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    print(f"[load_data] Loaded {len(df)} samples from {filepath}")
    return df


def generate_synthetic_data(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic training data that mimics stylistic differences
    between human-written and AI-generated text.

    Labels:
        0 = Human-written
        1 = AI-generated
    """
    rng = np.random.default_rng(random_state)

    human_texts = [
        "I honestly wasn't sure what to expect going in, but it blew me away.",
        "My grandma used to make this every Sunday and I still can't replicate it.",
        "Went for a walk this morning — cleared my head more than I thought it would.",
        "Not gonna lie, I've been putting this off for weeks and it shows.",
        "The movie was fine I guess, but the ending felt rushed.",
        "Tried a new coffee shop downtown. Honestly overpriced but the vibe was nice.",
        "Can someone explain why this keeps happening? Third time this week.",
        "I used to hate running but somehow it's become the one thing I look forward to.",
        "Some days you just wake up feeling off and there's no real reason why.",
        "Finished the book at 2am and immediately wanted to reread it.",
        "There's something weird about seeing your hometown change so much.",
        "The meeting dragged on for two hours and we solved exactly nothing.",
        "Cooking while stressed is weirdly therapeutic. Made pasta from scratch.",
        "Told myself I'd go to bed early. It's 1am. Classic.",
        "The new season started strong but episode 4 lost me completely.",
        "My dog has been extra clingy today and I'm not complaining.",
        "Finally fixed the bug that's been haunting me for three days. Pure relief.",
        "I thought I knew how to parallel park until I moved to this city.",
        "Weird how a song can immediately take you back five years.",
        "The presentation went okay. Could've been worse, could've been better.",
    ]

    ai_texts = [
        "Artificial intelligence has fundamentally transformed the landscape of modern computing by enabling systems to learn from data.",
        "In conclusion, it is evident that a multifaceted approach is required to address the complexities of this issue.",
        "The following sections provide a comprehensive overview of the key factors that contribute to this phenomenon.",
        "It is important to note that this methodology has several notable advantages over traditional approaches.",
        "Furthermore, the implementation of these strategies can lead to significant improvements in overall efficiency.",
        "This analysis demonstrates that the correlation between variables is both statistically significant and practically meaningful.",
        "The results clearly indicate that the proposed framework outperforms existing benchmarks across multiple dimensions.",
        "In summary, the evidence strongly supports the conclusion that this approach represents a viable solution.",
        "The data presented herein suggests a positive trend in the adoption of emerging technologies.",
        "Additionally, it should be acknowledged that there are inherent limitations to the scope of this study.",
        "The system leverages state-of-the-art machine learning techniques to deliver optimal performance outcomes.",
        "It is worth noting that the implications of these findings extend beyond the immediate context.",
        "The proposed solution offers a robust and scalable architecture for enterprise-level deployment.",
        "Through rigorous evaluation, it has been determined that the model achieves superior accuracy.",
        "The integration of advanced algorithms enables seamless processing of complex datasets.",
        "This document outlines the key components and their interactions within the proposed framework.",
        "The methodology employed in this investigation adheres to established scientific standards.",
        "It is crucial to emphasize the importance of continuous monitoring and iterative refinement.",
        "The comprehensive evaluation conducted reveals several important insights regarding performance.",
        "These findings have significant implications for the future development of AI-based systems.",
    ]

    records = []
    for i in range(n_samples // 2):
        base = human_texts[i % len(human_texts)]
        noise = " ".join(rng.choice(base.split(), size=min(3, len(base.split())), replace=True))
        text = base + " " + noise if rng.random() > 0.5 else base
        records.append({"text": text, "label": 0})

    for i in range(n_samples // 2):
        base = ai_texts[i % len(ai_texts)]
        noise = " ".join(rng.choice(base.split(), size=min(3, len(base.split())), replace=True))
        text = base + " " + noise if rng.random() > 0.5 else base
        records.append({"text": text, "label": 1})

    df = pd.DataFrame(records).sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"[load_data] Generated {len(df)} synthetic samples ({n_samples//2} human, {n_samples//2} AI)")
    return df


def load_data(source: str = "synthetic", n_samples: int = 2000) -> pd.DataFrame:
    """
    Main entry point for loading data.

    Args:
        source: 'synthetic' to generate data, or a file path to a CSV.
        n_samples: Number of samples to generate (only used for 'synthetic').
    """
    if source == "synthetic":
        return generate_synthetic_data(n_samples=n_samples)
    elif Path(source).exists():
        return load_from_csv(source)
    else:
        raise ValueError(f"Unknown source: '{source}'. Use 'synthetic' or a valid CSV path.")