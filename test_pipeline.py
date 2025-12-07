from main import answer_with_realtime_data

result = answer_with_realtime_data(
    folder_path="test_docs",
    json_path="Data/sensors/sensor_data_1.json",
    user_question="What is the biggest risk for my chickens right now?"
)

print(result["answer"])
