import json
import os
from pathlib import Path

# Path to the data folder
current_dir = Path(__file__).parent  # Gets the directory containing this script
data_folder = current_dir / "frontend" / "data"  # Navigate to frontend/data

data_folder.mkdir(parents=True, exist_ok=True)

# # Optional: Manually add categories for files (can leave empty if not needed)
# categories = {
#     "amazon.txt": "E-commerce",
#     "bmw.txt": "Automotive",
#     "costco.txt": "Retail",
#     "linkedin.txt": "Social Media",
#     # Add more as needed
# }


# Company display names
company_names = {
    "amazon": "Amazon",
    "apple": "Apple",
    "audi": "Audi",
    "belk": "Belk",
    "bestbuy": "Best Buy",
    "bjs": "BJ's",
    "bmw": "BMW",
    "burgerking": "Burger King",
    "burlington": "Burlington",
    "chickfila": "Chick-fil-A",
    "chipotle": "Chipotle",
    "costco": "Costco",
    "dillards": "Dillard's",
    "discord": "Discord",
    "dunkindonuts": "Dunkin' Donuts",
    "ford": "Ford",
    "freetaxusa": "FreeTax USA",
    "general_motors_gm": "General Motors (GM)",
    "googlemaps": "Google Maps",
    "homedepot": "Home Depot",
    "instagram": "Instagram",
    "jcpenny": "JC Penney",
    "kfc": "KFC",
    "linkedin": "LinkedIn",
    "lowes": "Lowe's",
    "lululemon": "Lululemon",
    "macys": "Macy's",
    "mcdonalds": "McDonald's",
    "mettler_toledo": "Mettler Toledo",
    "motorola": "Motorola",
    "officedepot": "Office Depot",
    "onstar": "OnStar",
    "qdoba": "Qdoba",
    "samsclub": "Sam's Club",
    "samsung": "Samsung",
    "shein": "SHEIN",
    "slack": "Slack",
    "starbucks": "Starbucks",
    "subway": "Subway",
    "tacobell": "Taco Bell",
    "tjmaxx": "TJ Maxx",
    "vivo": "Vivo",
    "zara": "Zara",
}


def format_company_name(filename):
    """
    Format the filename into a proper company display name.
    Args:
        filename (str): The original filename without .txt extension
    Returns:
        str: Properly formatted company name
    """
    # Remove .txt and any underscores
    base_name = filename.replace(".txt", "").lower()

    # Check if we have a custom mapping for this company
    if base_name in company_names:
        return company_names[base_name]

    # If no custom mapping, apply default formatting:
    # Split by underscores or hyphens and capitalize each word
    words = base_name.replace("-", " ").replace("_", " ").split()
    formatted_name = " ".join(word.capitalize() for word in words)

    return formatted_name


# Initialize metadata list
metadata_list = []

# Loop through all .txt files in the data folder
for filename in sorted(os.listdir(data_folder)):  # Sort filenames alphabetically
    if filename.endswith(".txt"):
        # Create metadata entry
        metadata = {
            "title": format_company_name(filename),
            "filename": filename,
            # "date_added": "",  # Optional: add date info here
            # "category": "",    # Optional: add category info here
            # "description": ""  # Optional: add description here
        }
        metadata_list.append(metadata)

# Sort metadata list by title
metadata_list.sort(key=lambda x: x["title"])

# Save metadata to a JSON file
output_path = data_folder / "metadata.json"
with open(output_path, "w", encoding="utf-8") as json_file:
    json.dump(metadata_list, json_file, indent=4, ensure_ascii=False)

print(f"Metadata saved to {output_path}")
print("\nProcessed companies:")
for metadata in metadata_list:
    print(f"- {metadata['title']}")
