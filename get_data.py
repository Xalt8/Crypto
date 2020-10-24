import requests
from datetime import datetime
import csv
import time

def stream_data():

    url = "https://api.coindcx.com/exchange/ticker"

    fields = ['market', 'change_24_hour', 'high', 'low', 'volume', 'last_price', 'bid', 'ask', 'timestamp']

    # Create a 'write-only' file writer object
    with open('data.csv', 'w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction='ignore', dialect='excel')
        csv_writer.writeheader() # Makes a header with the 'fields' values

    while True:
        # Append to write object
        with open('data.csv', 'a', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction='ignore', dialect='excel')
            
            response = requests.get(url)
            data = response.json()
                
            data_dict = data[0] # Get the values of the first entry in the list

            '''Check to see if the entry has been recorded'''
            last_entry = dict()
            if data_dict['timestamp'] != last_entry['timestamp']:
                last_entry = data_dict 
                csv_writer.writerow(last_entry)
                print(last_entry.values())
            else:
                continue
                
        time.sleep(1)


if __name__ == '__main__':
    stream_data()

        




