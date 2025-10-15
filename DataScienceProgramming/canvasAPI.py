import requests
import json
import csv

BASE_URL = "https://vinuni.instructure.com/api/v1"
TOKEN = "15876~3yEYenAtEZ4EHVu9FHDC8zTaPHwDCafDf48DEk98Y4V6em9yEwDJ4vwG4GRzQxtH"
COURSE_ID = "2819"  # Replace with actual course ID
headers = {"Authorization": f"Bearer {TOKEN}"}

# 2. Fetch assignments (handle pagination if needed)
assignments = []
url = f"{BASE_URL}/courses/{COURSE_ID}/assignments"

while url:
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    assignments.extend(data)

    # Handle pagination (Canvas includes 'next' in Link header)
    if 'next' in response.links:
        url = response.links['next']['url']
    else:
        url = None

# 3. Save as JSON
with open("assignments.json", "w", encoding="utf-8") as f:
    json.dump(assignments, f, indent=4)

print("✅ Saved assignments.json")

# 4. Save as CSV
csv_fields = ["id", "name", "due_at", "points_possible", "course_id"]
with open("assignments.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()
    for a in assignments:
        writer.writerow({
            "id": a.get("id"),
            "name": a.get("name"),
            "due_at": a.get("due_at"),
            "points_possible": a.get("points_possible"),
            "course_id": a.get("course_id"),
        })

print("✅ Saved assignments.csv")
