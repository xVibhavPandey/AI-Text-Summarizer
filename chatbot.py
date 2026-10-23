from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation",
    pipeline_kwargs=dict(
        temperature = 0.5,
        max_new_tokens= 100
    )
)
history = []

model = ChatHuggingFace(llm = llm)

while(True):
    user_input = input("You: ")
    history.append(user_input)
    if(user_input == "exit"):
        break
    result = model.invoke(history)
    history.append(result.content)
    print("AI: ", result.content)