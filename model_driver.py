import sys
import os
import json, datetime
import keras
from keras.callbacks import ModelCheckpoint
import data_gen

# hardcoded
from models.janknet.janknet_separation import JankNet

def main(argv):

    # Using argv for path names
    path_imap = argv[1]
    path_mmap = argv[2]
    history_path = argv[3]
    model_name = argv[4]

    if model_name == "janknet":
        net = JankNet()
    else:
        print(f"model name {model_name} not found")
        exit(-1)

    BATCH_SIZE = 64
    NUM_MMAPS_PER_IMAP = 5
    LEN_DATA = min(len(os.listdir(path_imap)), len(os.listdir(path_mmap)) * NUM_MMAPS_PER_IMAP)
    EPOCHS = 1

    curtime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    # checkpoint
    filepath= "./weights-janknet-{epoch:02d}-{loss:.2f}_" + curtime + ".hdf5"
    # save the minimum loss
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, save_best_only=False)
    callbacks_list = [checkpoint]
    # Fit the model
    history_obj = net.train(LEN_DATA, BATCH_SIZE, EPOCHS, data_gen.generator(path_imap, path_mmap, num_mmaps_per_imap=NUM_MMAPS_PER_IMAP), callbacks_list)
    # save the history object to a pickle file
    json.dump(history_obj.history, open(history_path + "_" + curtime, "w"))


if __name__ == "__main__":
    main(sys.argv)