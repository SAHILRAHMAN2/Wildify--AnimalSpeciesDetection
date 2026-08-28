import os
import json

TEST_DIR = r"C:\Users\sdrga\Downloads\test"

class_names = sorted([
    folder for folder in os.listdir(TEST_DIR)
    if os.path.isdir(os.path.join(TEST_DIR, folder))
])

with open("class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)

print("Total classes:", len(class_names))
print(class_names)
