import csv

def save_vehicle_counts(vehicle_counts, class_names, filename="vehicle_counts.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Vehicle Type", "Total Count"])
        for cls_id, count in vehicle_counts.items():
            writer.writerow([class_names[cls_id], count])
    print(f"Vehicle count data saved to {filename}.")
