**The DeepInstruments spec**
22 dec 2015

A. *Audio features*
    1. [DONE] Review silence detection.
    2. [DONE] Convert features to float32.
    3. Generate silenced frames in test set.
    4. [DONE] Review perceptual loudness reference in get_X

B. *Deep learning*
    1. [DONE] Write Graph model without Z
    2. [DONE] Make it a function in module "learning"
    3. [DONE] Solve core dump
    4. [DONE] Install bleeding-edge Keras
    5. [DONE] Train on categorical cross-entropy
    6. [DONE] Write data generator
    6. Add Z supervision

C. *Pitch supervision*
    1. [CLOSED] Get Gt samples for RWC
    2. [DONE] Check MIDI offsets in RWC dict
    3. [DONE] Write conversion from MIDI to ConvNet axis.

D. *Evaluation*
    1. [DONE] Write class-based accuracy measure
    2. [DONE] Write callbacks to monitor test error
    2. [DONE] Integrate the pipeline into a function so that the whole experiment can be ran in one step.
    3. [DONE] Measure class imbalance. How many decision windows per class ?
    4. Use MIR metrics for multi-label classif.
    5. [DONE] Make a 80/20 file-based split for the retained instruments.

E. *Display*
    1. Export filters from conv1 as images for the three experiments. Are they learned note models ?
    2. Make a figure for the architecture.

F. *Dataset*
1. [DONE] Get the full MedleyDB dataset
2. Update wrangling so that it lists files, not classes
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

G. Single-label classification
1. [DONE] Write get_activation
2. [DONE] Write get_indices (with boundary trimming)
3. [DONE] Write get_melody
4. [DONE] Memoize X with joblib
