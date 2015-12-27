**The DeepInstruments spec**
22 dec 2015

A. *Audio features*
1. [DONE] Review silence detection.
2. [DONE] Convert features to float32.
3. Generate silenced frames in test set.

B. *Deep learning*
1. [DONE] Write Graph model without Z
2. [DONE] Make it a function in module "learning"
3. [DONE] Solve floating-point error
4. [DONE] Install bleeding-edge Keras
5. Add Z supervision

C. *Pitch supervision*
1. Get Gt samples for RWC
2. Check MIDI offsets in RWC dict
3. Write conversion from MIDI to ConvNet axis.
4. Expand third dimension according to number of channels (link to ConvNet) ?

D. *Evaluation*
1. Write class-based accuracy measure
2. Integrate the pipeline into a function so that the whole experiment can be ran in one step.
3. Measure class imbalance. How many decision windows per class ?

E. *Display*
1. Export filters from conv1 as images for the three experiments. Are they learned note models ?
2. Make a figure for the architecture.
