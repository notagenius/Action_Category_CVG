   6332 0
   6919 1
    228 2
  11679 3
  16131 4
    756 5
  94847 6
  21017 7
 171437 8
 198253 9
 
 
527599


### NOTES
![](https://raw.githubusercontent.com/notagenius/Action_Category_CVG/master/model_tips.jpg?token=AGSPNMZSFEZD4NICGXVOBZS57TVYU)

### Dataset Visualization
- TSNE

### Future Checklist:
Visulization
- [ ] Confusion Matrix
- [ ] Skeleton & Predicted label
- [x] Accuracy curve

Model Candidates
- [x] LSTM / GRU 25 frames * 2 sample rate 
- [ ] TCN Non Causal largest I can fit, it is fully conventional
- [ ] TCN Causal 
- [ ] Bi-Directional RNN 
all models take the most simple structure no drop out
- [x] Fully Connected Layer

Evaluation Metrics
- [ ] Cross-Validation
- [x] S5 Only

### Even The Parameters 
to make the parameter number the same

### Find Out Where It Fails And Why and Give It A Solution

csv to HDF5 http://docs.h5py.org/en/latest/

refer: https://discuss.pytorch.org/t/how-to-speed-up-the-data-loader/13740/2

refer  LMDB (Lightning Memory-mapped Database)
