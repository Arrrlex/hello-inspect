from inspect_ai.scorer import mean, scorer
from inspect_ai.dataset import Sample

from inspect_ai import Task, task
from inspect_ai.scorer import Score, Target
from inspect_ai.solver import TaskState, basic_agent
from inspect_ai.tool import bash, python
from inspect_ai.util import sandbox


def load_dataset():
    prompt = "The file 'tables.py' contains a bug. Fix it.\nSubmit the fixed file as a Python script."

    five_times_table = "5 x 1 = 5\n5 x 2 = 10\n5 x 3 = 15\n5 x 4 = 20\n5 x 5 = 25"

    return [
        Sample(
            input=prompt,
            target=five_times_table,
            files={
                "tables.py": "assets/tables.py",
            },
        )
    ]


@scorer(metrics=[mean()])
def agent_eval_scorer():
    async def score(state: TaskState, target: Target):
        fixed_code = state.output.completion.strip()
        await sandbox().write_file("fixed_tables.py", fixed_code)
        output = await sandbox().exec(["python3", "fixed_tables.py"])
        return Score(
            value=target.text in output.stdout,
            answer=fixed_code,
            explanation=f"Output: {output.stdout}, Return code: {output.returncode}",
        )

    return score


@task
def agent_eval(max_attempts=3, message_limit=30):
    return Task(
        dataset=load_dataset(),
        solver=basic_agent(
            max_attempts=max_attempts,
            message_limit=message_limit,
            tools=[bash(), python()],
        ),
        scorer=agent_eval_scorer(),
        sandbox=("docker", "Dockerfile"),
    )
