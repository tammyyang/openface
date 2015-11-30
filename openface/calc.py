#!/usr/bin/env python2

import os
import sys
import time
import dlib
import cv2
import logging
from openface.alignment import NaiveDlib

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

class OFBase(object):
    def __init__(self):
        """Init function of OFBase"""

        self.align = NaiveDlib(args.dlibFaceMean, args.dlibFacePredictor) #FIXME
        return

    def getRep(self, imgPath, imgDim=96)
        """Get facenet representation

        @param imgPath: path of the target image

        """
        logging.debug("Processing {}.".format(imgPath))
        img = cv2.imread(imgPath)
        if img is None:
            raise Exception("Unable to load image: {}".format(imgPath))
        logging.debug("  + Original size: {}".format(img.shape))

        start = time.time()
        bb = self.align.getLargestFaceBoundingBox(img)
        if bb is None:
            raise Exception("Unable to find a face: {}".format(imgPath))
        logging.debug("  + Face detection took {} seconds.".format(time.time() - start))

        start = time.time()
        alignedFace = self.align.alignImg("affine", args.imgDim, img, bb)
        if alignedFace is None:
            raise Exception("Unable to align image: {}".format(imgPath))
        logging.debug("  + Face alignment took {} seconds.".format(time.time() - start))

        start = time.time()
        rep = self.net.forwardImage(alignedFace)
        logging.debug("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
        logging.debug("Representation:")
        logging.debug(rep)
        logging.debug("-----\n")
        return rep



parser = argparse.ArgumentParser()

parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
parser.add_argument('--dlibFaceMean', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "mean.csv"))
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

if args.verbose:
    print("Argument parsing and loading libraries took {} seconds.".format(
        time.time() - start))

start = time.time()
align = NaiveDlib(args.dlibFaceMean, args.dlibFacePredictor)
net = openface.TorchWrap(args.networkModel, imgDim=args.imgDim, cuda=args.cuda)
