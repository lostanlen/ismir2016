**The DeepInstruments spec**
22 dec 2015

A. *Audio features*
    1. [DONE] Review silence detection.
    2. [DONE] Convert features to float32.
    3. [CLOSED] Generate silenced frames in test set.
    4. [DONE] Review perceptual loudness reference in get_X
    5. [DONE] Explicitly ignore WavFileWarning
    6. [DONE] For standardization, only collect the mean and variance of activated frames

B. *Deep learning*
    1. [DONE] Write Graph model without Z
    2. [DONE] Make it a function in module "learning"
    3. [DONE] Solve core dump
    4. [DONE] Install bleeding-edge Keras
    5. [DONE] Train on categorical cross-entropy
    6. [DONE] Write data generator
    7. [DONE] Add Z supervision
    8. [DONE] Report mean and variance of loss after each epoch

C. *Pitch supervision*
    1. [CLOSED] Get Gt samples for RWC
    2. [DONE] Check MIDI offsets in RWC dict
    3. [DONE] Write conversion from MIDI to ConvNet axis.
    4. [DONE] Patch rankings for The Districts, Vermont
    5. [DONE] Extract Z
    6. [DONE] Extract G
    7. [DONE] Flowing Z and G in the datagen
    8. Max-pool G over the size of the decision length.
    9. [DONE] Write LambdaMerge function for difference
    10. [DONE] Define a tunable weight for the Z loss

D. *Evaluation*
    1. [DONE] Write class-based accuracy measure
    2. [DONE] Write callbacks to monitor test error
    2. [DONE] Integrate the pipeline into a function so that the whole experiment can be ran in one step.
    3. [DONE] Measure class imbalance. How many decision windows per class ?
    4. Use MIR metrics for multi-label classif.
    5. [DONE] Make a 80/20 file-based split for the retained instruments.

E. *Display*
    1. [DONE] Export filters from conv1 as images.
    2. [DONE] Make a figure for the architecture.
    3. [DONE] Make a figure for the duration of training set and test set for every instrument in single-label dataset.

F. *Dataset*
1. [DONE] Get the full MedleyDB dataset
2. [CLOSED] Update wrangling so that it lists files, not classes
3. [DONE] Restrict to a certain number of classes
4. [DONE] Take the max of stems activations that play the same instrument
5. [DONE] Write a function that outputs Y from the Medley instrument
   activations, called by generator
6. [DONE] Upload MedleyDB on di and cerfeuil
7. [DONE] Extract annotated vs non-annotated files for single-label classes
8. [DONE] If there are several stems of the same instrument in a given track,
   discard non-annotated stems from test set
9. [DONE] Separate singers between training set and test set to avoid artist bias
10. [DONE] Report misnomer of CroqueMadame_Pilot(Lakelot)_ACTIVATION_CONF.lab
11. [DONE] Make a patch script in __init_.py to handle all misnomers
12. [DONE] Discard overdrive, shoegaze, bleed and inactivity in clean electric guitar
13. [DONE] Use version control for the medleydb-single-instruments derived dataset
14. [DONE] Remove vocal FX tracks
15. [DONE] Remove first and last chunk (half-silent by definition) of every track

G. Single-label classification
1. [DONE] Write get_activation
2. [DONE] Write get_indices (with boundary trimming)
3. [DONE] Write get_melody
4. [DONE] Memoize training X with joblib
5. [DONE] Write a dedicated generator
6. [DONE] Standardize X in the generator
7. [DONE] Train deep neural network on X and Y
8. [DONE] Memoize test X with joblib
9. [DONE] Report class-wise accuracy with error bars

H.  Descriptors + Random forests baseline
1. [DONE] Compute MFCCs on the training data
2. [DONE] Also Delta and Delta-Delta MFCCs
3. [DONE] Also centroid, bandwidth, contrast, rolloff
4. [DONE] Generate half-overlapping chunks of X
5. [DONE] Summarize with mean and variance over chunks
6. [DONE] Generate Y's as integer classes
7. [DONE] Same in test set
8. [DONE] Run scikit-learn's random forest on it
9. [DONE] Report class-wise accuracy with error bars
10. [DONE] Discard clean guitar and male singer
11. [DONE] Bugfix half-overlapping chunks
12. [DONE] More Tp, Cl, Fl, Pn, and Vl examples (from solosDb)

I. Structured validation
1. [DONE] Extract the stem folder of each chunk path
2. [DONE] Assign votes to a dict where stems are keys
3. [DONE] Get the true class of each stem
4. [DONE] Write a systematic structured evaluator

J. Reproducibility
1. List all operations that are necessary
2. Review this list on gentiane

K. Scattering transform
1. [DONE] Write function get_paths in MATLAB
2. [DONE] Compute joint scattering features
3. Compute plain scattering features
4. Compute spiral scattering features
5. Review the importance of log compression
6. [DONE] Export in HDF5 from MATLAB to Python
7. Check that paths are ordered like in Python
8. Load HDF5, train RF, report accuracy

L. References
1. Fuhrmann: musical instrument classification
2. Joder et al. musical instrument classification
3. Dieleman and Benjamin deep learning for audio
4. Humphrey, Bello, and LeCun deep architectures for music informatics
5. Salamon and Bello : feature learning
6. [DONE] Li, Qian and Wang: ConvNets on raw audio for multilabel instrument recognition
7. [DONE] McFee et al. librosa
8. [DONE] Kingma & Ba: Adam optimizer
9. [DONE] Chollet: Keras package
10. [DONE] Bittner et al. MedleyDB
11. [CLOSED] Bruna, Szlam, LeCun. Learning Stable Group Invariant Representations.
12. [CLOSED] Mallat 2016. Understanding Deep Convolutional Networks.

M. 
