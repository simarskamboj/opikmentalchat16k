
# Configure your OpenAI API key
import os
os.environ["OPENAI_API_KEY"] = "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"  # Replace with your actual key
import os
import json
import datetime
from litellm.integrations.opik.opik import OpikLogger
import litellm
from litellm import (
    get_assistants,
    create_thread,
    add_message,
    run_thread_stream
)
from opik import track, opik_context

# ------------------------------------------------------------------------------
# Setup Environment Variables
# ------------------------------------------------------------------------------

# Initialize Opik Logger
opik_logger = OpikLogger()
litellm.callbacks = [opik_logger]

# ------------------------------------------------------------------------------
# Get an Existing Assistant
# ------------------------------------------------------------------------------
assistants = get_assistants(custom_llm_provider="openai")
if not assistants.data:
    raise ValueError("No assistants found. Please create one first.")
assistant = assistants.data[0]

print(f"Using assistant with id: {assistant.id}")

# ------------------------------------------------------------------------------
# Define Tracked LLM Chain Function
# ------------------------------------------------------------------------------
@track
def llm_chain(user_input):
    """
    Function to handle the user input and retrieve LLM response using streaming.
    - The @track decorator ensures Opik only logs 'user_input' as input and 'model_response' as output.
    """
    
    # Step 1: Create a conversation thread with the user's input
    thread = create_thread(
        custom_llm_provider="openai",
        messages=[{"role": "user", "content": user_input}]
    )
    print(f"Thread created with id: {thread.id}")

    # Optionally, add another message.
    #add_message(
    #    thread_id=thread.id,
    #    custom_llm_provider="openai",
    #    **{"role": "user", "content": "Also, can you provide a fun fact about AI?"}
    #)
    
    # Step 2: Start streaming the assistant's response
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
    # Update Span & Trace Separately (Without Affecting Opik's Input/Output Logging)
    # ------------------------------------------------------------------------------
    opik_context.update_current_trace(tags=["llm_chatbot"])  # Correctly adding trace tag


    return model_response  # 

# ------------------------------------------------------------------------------
# Example Execution
# ------------------------------------------------------------------------------
user_question = "Why is tracking and evaluation of LLMs important?"
final_response = llm_chain(user_question)
