### NOTES

![](https://raw.githubusercontent.com/notagenius/Action_Category_CVG/master/model_tips.jpg?token=AGSPNMZSFEZD4NICGXVOBZS57TVYU)
### Dataset Visualization
- TSNE

### Future Checklist:
Visulization
- [ ] Confusion Matrix
- [ ] Skeleton & Predicted label
- [ ] Accuracy curve

Model Candidates
- [ ] LSTM / GRU 25 frames * 2 sample rate 
- [ ] TCN Non Causal largest I can fit, it is fully conventional
- [ ] TCN Causal 
- [ ] Bi-Directional RNN 
all models take the most simple structure no drop out
- [ ] Fully Connected Layer

Evaluation Metrics
- [ ] Cross-Validation
- [ ] S5 Only

### even the Parameters 
to make the parameter number the same

### find out where it fails and why and give a solution

csv to HDF5
http://docs.h5py.org/en/latest/
refer: https://discuss.pytorch.org/t/how-to-speed-up-the-data-loader/13740/2
refer  LMDB (Lightning Memory-mapped Database)
