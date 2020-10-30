import requests
from datetime import datetime
import csv
import time

def stream_data():
    ''' Function used to get the ticker data for BTC/INR from coindcx.com's website through their API
        and appends the data to a CSV file on disk '''

    url = "https://api.coindcx.com/exchange/ticker"

    fields = ['market', 'change_24_hour', 'high', 'low', 'volume', 'last_price', 'bid', 'ask', 'timestamp']

    # Create a 'write-only' file writer object
    with open('data.csv', 'w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction='ignore', dialect='excel')
        csv_writer.writeheader() # Makes a header with the 'fields' values

    '''Check to see if the entry has been recorded'''
    last_entry = dict()
    
    while True:
        # Append to writer object
        with open('data.csv', 'a', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction='ignore', dialect='excel')
            
            response = requests.get(url)
            data = response.json()
                
            data_dict = data[0] # Get the values of the first entry in the list (BTC/INR)

            if data_dict == last_entry: # Checks to see if the data that is going to be added is already there
                continue # If the entry is already there - skip adding it
            else:
                last_entry = data_dict # Update the last entry record
                csv_writer.writerow(last_entry) # Write to the file
                print(last_entry.values()) # Prints out the entry to the consol
                
                
        time.sleep(1)


if __name__ == '__main__':
    stream_data()

        




