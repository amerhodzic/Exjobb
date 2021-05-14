import json

kupor = ['Beehive-1', 'Beehive-2', 'Beehive-3', 'Beehive-4']
num_of_features = [13, 40]


for kupa in kupor:
    for j in num_of_features:
        dest = f'/Users/amerhodzic/Documents/Exjobb/Ljud/{kupa}/split_by_15_minutes/results/Mfcc-{j}-dict.json'

        with open(dest) as json_file:
            data = json.load(json_file)

        Queen = []
        Brom = []
        for key in data:
            if 'shifted' in str(key):
                Queen.append(data[key])
            elif 'Brom' in key:
                Brom.append(data[key])
        print(kupa)
        print(f'Number of features = {j}\n')
        print('Queen data')
        for i in set(Queen):
            print(f'\tNumber in cluster {i} = {Queen.count(i)}')
        print()
        print('Brom data')
        for i in set(Brom):
            print(f'\tNumber in cluster {i} = {Brom.count(i)}')