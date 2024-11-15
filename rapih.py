import json

def beautify_json(input_file, output_file=None, indent=4):
    """
    Beautify a JSON file with indentation for better readability.
    
    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output JSON file. If None, it overwrites the input file.
        indent (int): Number of spaces to use as indentation.
        
    Returns:
        None
    """
    # Load JSON data from file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Define the output file
    if output_file is None:
        output_file = input_file
    
    # Write beautified JSON to the output file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=indent)
    
    print(f"Beautified JSON saved to {output_file}")

# Example usage:
beautify_json("path/to/input.json", "path/to/output.json")
