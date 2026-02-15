import xml.etree.ElementTree as ET
import json
from pathlib import Path
from tqdm import tqdm

# ====================================================================
# 1. CONFIGURATION
# ====================================================================

# --- Input Paths ---
IDD_XML_ROOT = Path("C:/Users//Documents/Hackathon_ADAS/IDD117K_Detection/IDD_Detection/Annotations")
IDD95K_JSON_ROOT = Path("C:/Users//Documents/Hackathon_ADAS/IDD117K_Detection/IDD_95kDetection")

# --- Classes to REMOVE ---
# This is a "discard list". Any object with a class name found in this list will be removed.
CLASSES_TO_REMOVE = {
    # Critically rare classes
    "trailer",
    "train",
    "caravan",
    
    # Ambiguous or irrelevant classes
    "ego vehicle",
    "pole",
    "vehicle fallback",
    
    # Add any other classes you decide to remove here
}

# --- Safety Switch ---
# Set to True to create a new "cleaned" directory.
# Set to False to overwrite the original files (use with caution!).
# RECOMMENDATION: Keep this as True.
CREATE_BACKUP = True

# ====================================================================
# 2. THE CLEANING SCRIPT
# ====================================================================

def clean_xml_file(xml_path, classes_to_remove):
    """Reads an XML file, removes objects with unwanted classes, and saves it."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        objects_to_remove = []
        for obj in root.findall('object'):
            name_tag = obj.find('name')
            if name_tag is not None and name_tag.text and name_tag.text.strip() in classes_to_remove:
                objects_to_remove.append(obj)
        
        # If we found objects to remove, modify the tree
        if objects_to_remove:
            for obj in objects_to_remove:
                root.remove(obj)
            tree.write(xml_path)
            return True # Indicates the file was modified
        return False # Indicates no changes were made

    except ET.ParseError:
        print(f"\n⚠️ Warning: Skipping malformed XML file: {xml_path}")
        return False

def clean_json_file(json_path, classes_to_remove):
    """Reads a JSON file, removes objects with unwanted classes, and saves it."""
    try:
        with open(json_path, 'r') as f:
            # The JSON file is a list of objects
            data = json.load(f)

        # Filter the list, keeping only objects whose 'name' is NOT in the removal set
        original_count = len(data)
        cleaned_data = [obj for obj in data if obj.get("name") and obj.get("name").strip() not in classes_to_remove]
        
        # If the list was modified, write the new data back to the file
        if len(cleaned_data) < original_count:
            with open(json_path, 'w') as f:
                json.dump(cleaned_data, f, indent=2)
            return True # Indicates the file was modified
        return False # Indicates no changes were made

    except (json.JSONDecodeError, AttributeError, TypeError):
        print(f"\n⚠️ Warning: Skipping malformed JSON file: {json_path}")
        return False

def main():
    """Main function to run the cleaning process."""
    
    # --- Handle XML Cleaning ---
    print("--- Starting XML Cleaning Process ---")
    if CREATE_BACKUP:
        print(f"SAFE MODE: A backup of your XMLs will be created at '{str(IDD_XML_ROOT.parent / (IDD_XML_ROOT.name + '_Original'))}'")
        shutil.copytree(IDD_XML_ROOT, IDD_XML_ROOT.parent / (IDD_XML_ROOT.name + '_Original'), dirs_exist_ok=True)
    
    xml_files = list(IDD_XML_ROOT.rglob('*.xml'))
    modified_xml_count = 0
    for file in tqdm(xml_files, desc="Cleaning XML files"):
        if clean_xml_file(file, CLASSES_TO_REMOVE):
            modified_xml_count += 1
    print(f"✅ XML Cleaning Complete. {modified_xml_count} files were modified.")

    # --- Handle JSON Cleaning ---
    print("\n--- Starting JSON Cleaning Process ---")
    if CREATE_BACKUP:
        print(f"SAFE MODE: A backup of your JSONs will be created at '{str(IDD95K_JSON_ROOT.parent / (IDD95K_JSON_ROOT.name + '_Original'))}'")
        shutil.copytree(IDD95K_JSON_ROOT, IDD95K_JSON_ROOT.parent / (IDD95K_JSON_ROOT.name + '_Original'), dirs_exist_ok=True)

    json_files = list(IDD95K_JSON_ROOT.rglob('*.json'))
    modified_json_count = 0
    for file in tqdm(json_files, desc="Cleaning JSON files"):
        if clean_json_file(file, CLASSES_TO_REMOVE):
            modified_json_count += 1
    print(f"✅ JSON Cleaning Complete. {modified_json_count} files were modified.")

    print("\n" + "="*60)
    print("🎉 All raw annotation files have been cleaned!")
    print("You are now ready to run your conversion script on the cleaned data.")
    print("="*60)


if __name__ == "__main__":
    import shutil # Import shutil here as it's only used in main
    main()
    input("\nPress Enter to exit...")

