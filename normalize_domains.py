import sqlite3

# Connect to DB
conn = sqlite3.connect("resume_data.db")
cursor = conn.cursor()

# Define normalization map
normalization_map = {
    "AI / Machine Learning": "AI/ML",
    "Artificial Intelligence": "AI/ML",
    "Machine Learning": "AI/ML",
    "DevOps / Infrastructure": "DevOps & Infrastructure",
    "Cloud Engineering": "Cloud & DevOps",
    "General Software Engineering": "Software Engineering"
}

# Loop through normalization map and update database
for old_val, new_val in normalization_map.items():
    cursor.execute("UPDATE candidates SET domain = ? WHERE domain = ?", (new_val, old_val))

conn.commit()
conn.close()
print("✅ Existing domains normalized in the database.")
