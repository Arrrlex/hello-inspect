from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice


@task
def gpqa_diamond(cot: bool = True) -> Task:
    return Task(
        dataset=hf_dataset(
            "Idavidrein/gpqa",
            name="gpqa_diamond",
            split="train",
            sample_fields=record_to_sample,
            limit=10,
        ),
        solver=multiple_choice(shuffle=True, cot=cot),
        scorer=choice(),
        config=GenerateConfig(temperature=0.5),
    )


# map records to inspect samples (note that target is always "A" in the,
# dataset, we will shuffle the presentation of options to mitigate this)
def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["Question"],
        choices=[
            str(record["Correct Answer"]),
            str(record["Incorrect Answer 1"]),
            str(record["Incorrect Answer 2"]),
            str(record["Incorrect Answer 3"]),
        ],
        target="A",
        id=record["Record ID"],
    )
