from opik import Opik, track, opik_context
from opik.evaluation import evaluate
from opik.evaluation.metrics import (AnswerRelevance, Equals, LevenshteinRatio, Hallucination, Moderation, ContextRecall, ContextPrecision)
from opik.integrations.openai import track_openai
import openai
import os
from litellm.integrations.opik.opik import OpikLogger
import litellm
from litellm import (
    get_assistants,
    create_thread,
    run_thread_stream
)

# ------------------------------------------------------------------------------
# Setup OpenAI and Opik Integration
# ------------------------------------------------------------------------------
# os.environ["OPENAI_API_KEY"] = ""  # Replace with your actual key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

opik_logger = OpikLogger()
litellm.callbacks = [opik_logger]

MODEL = "gpt-3.5-turbo"
standard_gpt_wo_base =  "asst_ti5mnO64hx3u18kZdcnljYeU"
standard_chatgpt =  'asst_Qjj0WoZDhnFqeYV2DEdUVRAU'
usmle =  'asst_4zVTKy9jl6tddcAWMZNWW3HK'
msl = 'asst_PsaE2upQI2K7wWmyzZ73A8Gp'
padcev = 'asst_CCIXbmGWgAjT2MscOMpBFphu'
# ------------------------------------------------------------------------------
# Get an Existing Assistant
# ------------------------------------------------------------------------------
def select_assistant_by_id(assistants, target_id):
    for assistant in assistants.data:
        if assistant.id == target_id:
            return assistant
    raise ValueError(f"Assistant with ID {target_id} not found.")

# Fetch available assistants
assistants = get_assistants(custom_llm_provider="openai")

# Define the target assistant ID
target_assistant_id = standard_gpt_wo_base  # Change this ID as needed

# Select the assistant
assistant = select_assistant_by_id(assistants, target_assistant_id)

print(f"Using assistant with id: {assistant.id}")

print(f"Using assistant with id: {assistant.id}")

# ------------------------------------------------------------------------------
# Define Tracked LLM Application Function
# ------------------------------------------------------------------------------
@track
def your_llm_application(user_input: str) -> str:
    """
    Function to handle user input and retrieve LLM response using streaming.
    Logs only 'user_input' as input and 'model_response' as output.
    """
    thread = create_thread(
        custom_llm_provider="openai",
        messages=[{"role": "user", "content": user_input}]
    )
    print(f"Thread created with id: {thread.id}")

    print("Starting streaming response...")
    stream = run_thread_stream(
        custom_llm_provider="openai",
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    model_response = ""

    with stream as streamer:
        for chunk in streamer:
            if hasattr(chunk, "data") and hasattr(chunk.data, "delta"):
                delta_content = getattr(chunk.data.delta, "content", [])
                for block in delta_content:
                    if hasattr(block, "text") and hasattr(block.text, "value"):
                        model_response += block.text.value  # Accumulate streamed response

        streamer.until_done()

    print("Streaming complete.")
    print("Final output:", model_response)

    # ------------------------------------------------------------------------------
    # Update Span & Trace Separately
    # ------------------------------------------------------------------------------
    opik_context.update_current_trace(tags=["open AI assistant", target_assistant_id])

    opik_context.update_current_span(
        name="llm_chain",
        metadata={"user_input": user_input, "model_response": model_response}
    )

    return model_response  # Ensure only the model output is returned

# ------------------------------------------------------------------------------
# Define the Evaluation Task
# ------------------------------------------------------------------------------
def evaluation_task(x):
    return {
        "output": your_llm_application(x['input'])
    }

# ------------------------------------------------------------------------------
# Fetch Dataset
# ------------------------------------------------------------------------------
client = Opik()
dataset = client.get_dataset(name="PadcevQAMC")  # Using your dataset name

# ------------------------------------------------------------------------------
# Define and Run Evaluation
# ------------------------------------------------------------------------------
metrics = [Hallucination(), Moderation(), AnswerRelevance(), ContextRecall(), ContextPrecision()]
metrics = [Hallucination(), AnswerRelevance(), Equals(), LevenshteinRatio()]

evaluation = evaluate(
    dataset=dataset,
    task=evaluation_task,
    scoring_metrics=metrics,
    experiment_config={
        "model": MODEL
    }
)

print("Evaluation Complete!")
