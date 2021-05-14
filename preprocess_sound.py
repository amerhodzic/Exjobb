from shutil import copy2
from glob import glob
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
from Split_audio import SplitWavAudioMubin
import librosa, time, numpy as np, sys, os, json
from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
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
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=number_of_features).T,axis=0)
        mfcc_list.append(mfccs)
    end = time.time()
    tid = end - start
    print(f'Feature Extraction Ended \nTime used: {tid} seconds')
    np.set_printoptions(threshold=sys.maxsize)
    mfcc_arr = np.array(mfcc_list)
    kupa = source_folder.split('/')[-3]
    output_file = f'Mfcc_{number_of_features}_{kupa}.out'
    np.savetxt(output_file, mfcc_arr)
    return output_file

def compute_clustering(feature_file, sound_files_dest, number_of_features, epsilon, min_samp):
    dim_red = MDS(n_components=3, random_state=2)
    X = np.loadtxt(feature_file)
    audio_files = sorted(glob(sound_files_dest + '*.wav'))
    print(np.shape(X))
    labeled_files = {}
    X = StandardScaler().fit_transform(X)
    two_dim_obj = dim_red.fit_transform(X)
    db = DBSCAN(eps=epsilon, min_samples=min_samp).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels))
    for i in set(labels):
        print(f'Number of points in cluster {i} => {list(labels).count(i)}')
    n_noise_ = list(labels).count(-1)
    print(f'Percent of noise = {round(n_noise_ / len(labels), 2)}%')
    for k in range(0, len(labels), 1):
        labeled_files[audio_files[k]] = str(labels[k])
    json_file = json.dumps(labeled_files, indent=4)
    f = open(f"{sound_files_dest}results/Mfcc-{number_of_features}-dict.json", "w")
    f.write(json_file)
    f.close()
    two_dim_obj_df = pd.DataFrame({'X': two_dim_obj[:, 0],
                            'Y': two_dim_obj[:, 1],
                            'Z': two_dim_obj[:, 2],
                            'digit': labels})
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.grid(b=True, color='grey',
            linestyle='-.', linewidth=0.3,
            alpha=0.2)
    my_cmap = plt.get_cmap('tab20')
    scatter = ax.scatter3D(two_dim_obj_df['X'], two_dim_obj_df['Y'], two_dim_obj_df['Z'], c=labels, cmap=my_cmap)
    ax.legend(*scatter.legend_elements(), title="Clusters", loc='best')
    plt.title(label=f'Epsilon={epsilon}, Min_samples={min_samp}, Mfcc_{number_of_features}_features')
    plt.savefig(f'{sound_files_dest}results/Mfcc_{number_of_features}-epsilon={epsilon}-min_samples={min_samp}.png')
    plt.close()

def number_of_features(feature_file):
    X = np.loadtxt(feature_file)
    print(np.shape(X))

def find_epsilon(feature_file, min_samp, kupa, number_of_features, dest):
    X = np.loadtxt(feature_file)
    X = StandardScaler().fit_transform(X)
    nearest_neighbors = NearestNeighbors(n_neighbors=min_samp + 1)
    neighbors = nearest_neighbors.fit(X)
    distances, indices = neighbors.kneighbors(X)
    distances = np.sort(distances[:, min_samp], axis=0)
    i = np.arange(len(distances))
    knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
    epsilon = distances[knee.knee]
    fig = plt.figure(figsize=(5, 5))
    knee.plot_knee()
    plt.title(f"Elbow for {kupa}\nMfcc-{number_of_features}-features\nEpsilon={epsilon}")
    plt.xlabel("Points")
    plt.ylabel("Distance")
    plt.savefig(f'{dest}results/Elbow_{kupa}_Mfcc_{number_of_features}_features.png')
    plt.close()
    print(epsilon)
    return epsilon

kupor = ['Beehive-1', 'Beehive-2', 'Beehive-3', 'Beehive-4']
min_samples=5
num_of_features = [13, 40]
for kupa in kupor:
    for j in num_of_features:
        destination = f'/Users/amerhodzic/Documents/Exjobb/Ljud/{kupa}/split_by_15_minutes/'
        print(f'{kupa}, Number of features = {j}')
        features = extract_mfcc_feature_file(source_folder=destination, number_of_features=j)
        eps = find_epsilon(f'Mfcc_{j}_{kupa}.out', min_samp=min_samples, kupa=kupa, number_of_features=j, dest=destination)
        compute_clustering(feature_file=f'Mfcc_{j}_{kupa}.out', sound_files_dest=destination, number_of_features=j, epsilon=eps, min_samp=min_samples)


#    #compute_clustering(feature_file='/Users/amerhodzic/PycharmProjects/pythonProject/ljud_1/split_by_10_minutes/Mfcc_13.out', sound_files_dest='/Users/amerhodzic/PycharmProjects/pythonProject/ljud_1/split_by_10_minutes/', number_of_features=13)