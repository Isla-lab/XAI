import os
import re
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def normalize_rule(rule):
    rule = re.sub(r',V\d+', '', rule)  
    rule = re.sub(r'V\d+,', '', rule)  
    rule = re.sub(r'dir\([^)]*\)', '', rule)  
    rule = re.sub(r'ranges_dist\([^)]*\)', '', rule) 
    rule = re.sub(',', '', rule)
    return rule.strip()

def extract_unique_rules(rules):
    unique_rules = set()
    for rule in rules:
        normalized_rule = normalize_rule(rule)
        if normalized_rule:
            unique_rules.add(normalized_rule)
    return list(unique_rules)

def hamming_distance(rules1, rules2):
    normalized_rules1 = set(sorted(extract_unique_rules(rules1)))
    normalized_rules2 = set(sorted(extract_unique_rules(rules2)))
    distance = len(normalized_rules1.difference(normalized_rules2))
    print('\n\n\n')
    print(rules1)
    print(rules2)
    print(normalized_rules1.difference(normalized_rules2))
    return distance

def read_rules_from_file(filepath):
    with open(filepath, 'r') as file:
        rules = file.readlines()
    return [rule.strip() for rule in rules]

def calculate_distances_in_directory(directory_path):
    distances = []
    # Get all files matching the pattern and sort by the numeric part in their name
    file_names = [file for file in os.listdir(directory_path)]
    
    previous_rules = None
    for filename in file_names:
        filepath = os.path.join(directory_path, filename)
        current_rules = read_rules_from_file(filepath)
        if previous_rules is not None:
            distance = hamming_distance(previous_rules, current_rules)
            distances.append(distance)
        previous_rules = current_rules

    return distances

# Define paths
script_path = os.path.dirname(os.path.realpath(__file__))
directory_path = os.path.join(script_path, "rules")

# Calculate distances between consecutive files
distances = calculate_distances_in_directory(directory_path)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(range(2, 2 + len(distances)), distances, marker='o', linestyle='-')
plt.xlabel('Batch of extraction')
plt.ylabel('Hamming Distance')

# Customize X-axis ticks
xticks = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
plt.xticks(xticks)

# Add grid and show plot
plt.grid(True)
plt.savefig(f"{script_path}/hamming_dist.png")
