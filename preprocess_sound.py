from shutil import copy2
from glob import glob
from Split_audio import SplitWavAudioMubin
import librosa, time, numpy as np, sys, os, json
from sklearn.cluster import DBSCAN
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def copy_selected_files(source_folder, minutes_beetween_files):
    dest = f'{source_folder}split_by_{minutes_beetween_files}_minutes/'
    try:
        os.mkdir(dest)
    except OSError as error:
        print(error)
    audio_files = sorted(glob(source_folder + '*.wav'))
    for i in range(0, len(audio_files), minutes_beetween_files * 2):
        copy2(audio_files[i], dest)
    return dest


def split_sound_files(source_folder, seconds_per_split):
    files = glob(source_folder + '*.wav')
    for file in files:
        split_wav = SplitWavAudioMubin(source_folder, file)
        split_wav.multiple_split(sec_per_split=seconds_per_split)
        os.remove(file)

def extract_mfcc_feature_file(source_folder, number_of_features):
    audio_files = sorted(glob(source_folder + '*.wav'))
    print('Feature Extraction Started')
    start = time.time()
    mfcc_list = []
    for file in range(0, len(audio_files), 1):
        X, sample_rate = librosa.load(audio_files[file], sr=8000)
        if '2021' in (audio_files[file].split('/')[1].split('-')):
            X = X[np.arange(5512, len(X) - 5512)]
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=number_of_features, hop_length=200).T,axis=0)
        mfcc_list.append(mfccs)
    end = time.time()
    tid = end - start
    print(f'Feature Extraction Ended \nTime used: {tid} seconds')
    np.set_printoptions(threshold=sys.maxsize)
    mfcc_arr = np.array(mfcc_list)
    output_file = f'{source_folder}Mfcc_{number_of_features}.out'
    np.savetxt(output_file, mfcc_arr)
    return output_file

def compute_clustering(feature_file, sound_files_dest, number_of_features):
    tsne = TSNE(n_components=2, random_state=10)
    X = np.loadtxt(feature_file)
    audio_files = sorted(glob(sound_files_dest + '*.wav'))
    labeled_files = {}
    X = StandardScaler().fit_transform(X)
    tsne_obj = tsne.fit_transform(X)
    for i in np.arange(5, 5.1, 0.2):
        for j in range(4, 5, 1):
            db = DBSCAN(eps=i, min_samples=j).fit(tsne_obj)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels))
            n_noise_ = list(labels).count(-1)
            tsne_df = pd.DataFrame({'X': tsne_obj[:, 0],
                                    'Y': tsne_obj[:, 1],
                                    'digit': labels})
            for k in range(0, len(labels), 1):
                labeled_files[audio_files[k]] = str(labels[k])
            json_file = json.dumps(labeled_files, indent=4)
            f = open(f"{sound_files_dest}dict.json", "w")
            f.write(json_file)
            f.close()
            sns.scatterplot(x='X', y='Y', data=tsne_df, hue='digit', palette='deep')
            plt.title(label=f'Epsilon={i}, Min_samples={j}, Mfcc_{number_of_features}')
            plt.savefig(f'{sound_files_dest}Mfcc_{number_of_features}-epsilon={i}-min_samples={j}.png')
            plt.close()



#destination = copy_selected_files(source_folder='/Users/amerhodzic/PycharmProjects/pythonProject/ljud_1/', minutes_beetween_files=10)
#split_sound_files(source_folder=destination, seconds_per_split=5)
#features = extract_mfcc_feature_file(source_folder=destination, number_of_features=13)
#compute_clustering(feature_file=features, sound_files_dest=destination, number_of_features=13)
compute_clustering(feature_file='/Users/amerhodzic/PycharmProjects/pythonProject/ljud_1/split_by_10_minutes/Mfcc_13.out', sound_files_dest='/Users/amerhodzic/PycharmProjects/pythonProject/ljud_1/split_by_10_minutes/', number_of_features=13)