from pydub import AudioSegment
import math


class SplitWavAudioMubin():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename.split('/')[-1]
        self.filepath = filename
        self.audio = AudioSegment.from_wav(self.filepath)

    def get_duration(self):
        return self.audio.duration_seconds

    def single_split(self, from_min, to_min, split_filename):
        t1 = from_min * 1000
        t2 = to_min * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(f'{self.folder}{split_filename}', format="wav")

    def multiple_split(self, sec_per_split):
        total_sec = math.ceil(self.get_duration())
        for i in range(5, total_sec, sec_per_split):
            split_fn = (f'_{str(i)}.').join(self.filename.split('.'))
            self.single_split(i, i + sec_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_sec - sec_per_split:
                print('All splited successfully')