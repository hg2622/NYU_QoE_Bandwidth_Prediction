import re
import pandas as pd
from google.colab import drive

# Mount Google Drive to access files
drive.mount('/content/drive')

# Define the function to process the log file
def extract_bitrate(log_file, output_csv):
    # Read the log file
    with open(log_file, 'r') as file:
        log_data = file.read()

    # Regex to extract bitrate data
    pattern = r"\[\s*\d+\]\s+[\d\.]+-[\d\.]+\s+sec\s+[\d\.]+\s+(KBytes|MBytes)\s+([\d\.]+)\s+Mbits/sec"
    matches = re.findall(pattern, log_data)

    # Process data and convert KBytes to MBytes
    data = []
    for match in matches:
        unit, bitrate = match
        bitrate = float(bitrate)
        data.append(bitrate)

    # Create a dataframe
    df = pd.DataFrame(data, columns=['Bitrate (MBytes/sec)'])
    
    # Print first 5 and last 5 entries
    print("First 5 entries:")
    print(df.head())
    print("\nLast 5 entries:")
    print(df.tail())
    
    # Truncate to first 1294 rows as per your requirement
    df = df[:1294]
    
    # Save the dataframe to CSV
    df.to_csv(output_csv, index=False)

# Example usage
log_file = '/content/drive/My Drive/9.10.1'  # Update with the correct path to your log file
output_csv = '/content/drive/My Drive/9.10.1_csv.csv'  # CSV file to be saved

# Run the extraction
extract_bitrate(log_file, output_csv)
