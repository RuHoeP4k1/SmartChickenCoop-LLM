from main import answer_with_realtime_data

import os
print("Working directory:", os.getcwd())
print("Exists?", os.path.exists("test_docs"))
"""

result = answer_with_realtime_data(
    folder_path="test_docs",
    json_path="Data/sensors/sensor_data_1.json",
    user_question="What is the biggest risk for my chickens right now?"
)

print(result["answer"])"""
