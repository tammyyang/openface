#!/usr/bin/env python2
#
# Example to classify faces.
# Brandon Amos
# 2015/10/11
#
# Copyright 2015 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import cv2
import itertools
import os
import pickle

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, ".."))

import openface
import openface.helper
from openface.data import iterImgs

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC

from tools import plot

modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


def getRep(imgPath):
    img = cv2.imread(imgPath)
    if img is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    if args.verbose:
        print("  + Original size: {}".format(img.shape))

    bb = align.getLargestFaceBoundingBox(img)
    if bb is None:
        raise Exception("Unable to find a face: {}".format(imgPath))

    alignedFace = align.alignImg("affine", args.imgDim, img, bb)
    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))

    rep = net.forwardImage(alignedFace)
    return rep


def train(args):
    print("Loading embeddings.")
    fname = "{}/labels.csv".format(args.workDir)
    labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory.
    fname = "{}/reps.csv".format(args.workDir)
    embeddings = pd.read_csv(fname, header=None).as_matrix()
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)

    param_grid = [
        {'C': [1, 10, 100, 1000],
            'kernel': ['linear']},
        {'C': [1, 10, 100, 1000],
            'gamma': [0.001, 0.0001],
            'kernel': ['rbf']}
    ]
    svm = GridSearchCV(
        SVC(probability=True),
        param_grid, verbose=4, cv=5, n_jobs=1
    ).fit(embeddings, labelsNum)
    print("Best estimator: {}".format(svm.best_estimator_))
    print("Best score on left out data: {:.2f}".format(svm.best_score_))

    with open("{}/classifier.pkl".format(args.workDir), 'w') as f:
        pickle.dump((le, svm), f)


def predict(le, svm, imgpath):
    try:
        rep = getRep(imgpath)
        return svm.predict_proba(rep)[0]
    except:
        print("Cannot predict %s successfully" %imgpath)
        return None

def check_ext(file_name, extensions):
    if file_name.endswith(extensions):
        return True
    return False

def find_images(dir_path):
    output = []
    for dirPath, dirNames, fileNames in os.walk(dir_path):
        for f in fileNames:
            if check_ext(f, ('.jpg', 'png')):
                output.append(os.path.join(dirPath, f))
    return output

def check_path(path):
    if os.path.isfile(path):
        if check_ext(path, ('.jpg', 'png')):
            return [path]
    elif os.path.isdir(path):
        return find_images(path)

def find_person_from_path(path):
    person = os.path.basename(os.path.dirname(path))
    return int(person.replace('person-', ''))

def print_output(output, accu):
    cm = []
    for p in sorted(output.keys()):
        ma = np.array(output[p])
        cm.append(ma.mean(0))
        if accu[p][0] == 0:
            perc = 0
        else:
            perc = float(accu[p][1])/float(accu[p][0])
        print("person-%i can be identified with %.2f accuracy"
              %(p, perc))
    plot.PLOT().plot_confusion_matrix(np.array(cm))

def infer(args):
    output = {}
    accu = {}
    with open(args.classifierModel, 'r') as f:
        (le, svm) = pickle.load(f)
        for f in check_path(args.img):
            true_p = find_person_from_path(f)
            if true_p not in output:
                output[true_p] = []
            if true_p not in accu:
                accu[true_p] = [0, 0]
            accu[true_p][0] += 1
            predictions = predict(le, svm, f)
            maxI = np.argmax(predictions)
            person = le.inverse_transform(maxI)
            person = int(person.replace('person-', ''))
            if predictions is not None:
                output[true_p].append(predictions)
            if person == true_p:
                accu[true_p][1] += 1
            #break
    print_output(output, accu)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dlibFaceMean', type=str,
                        help="Path to dlib's face predictor.",
                        default=os.path.join(dlibModelDir, "mean.csv"))
    parser.add_argument('--dlibFacePredictor', type=str,
                        help="Path to dlib's face predictor.",
                        default=os.path.join(dlibModelDir,
                                             "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument('--dlibRoot', type=str,
                        default=os.path.expanduser(
                            "~/src/dlib-18.16/python_examples"),
                        help="dlib directory with the dlib.so Python library.")
    parser.add_argument('--networkModel', type=str,
                        help="Path to Torch network model.",
                        default=os.path.join(openfaceModelDir, 'nn4.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    subparsers = parser.add_subparsers(dest='mode', help="Mode")
    trainParser = subparsers.add_parser('train',
                                        help="Train a new classifier.")
    trainParser.add_argument('workDir', type=str,
                             help="The input work directory containing 'reps.csv' and 'labels.csv'. Obtained from aligning a directory with 'align-dlib' and getting the representations with 'batch-represent'.")

    inferParser = subparsers.add_parser('infer',
                                        help='Predict who an image contains from a trained classifier.')
    inferParser.add_argument('classifierModel', type=str)
    inferParser.add_argument('img', type=str,
                             help="Input image.")

    args = parser.parse_args()

    sys.path.append(args.dlibRoot)
    import dlib
    from openface.alignment import NaiveDlib  # Depends on dlib.

    align = NaiveDlib(args.dlibFaceMean, args.dlibFacePredictor)
    net = openface.TorchWrap(
        args.networkModel, imgDim=args.imgDim, cuda=args.cuda)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'infer':
        infer(args)
