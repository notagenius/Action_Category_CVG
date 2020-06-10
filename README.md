![repo not ready](https://raw.githubusercontent.com/notagenius/openTMAS/master/docs/under_construction_title_card.png)

### Frames Calculation

Toatal Frames: 527599

| Label | Count |
| ----- | ----- |
| 0  | 6332  |
| 1  | 6919  |
| 2  | 228   |
| 3  | 11679 |
| 4  | 16131 |
| 5  | 756   |
| 6  | 94847 |
| 7  | 21017 |
| 8  | 171437|
| 9  | 198253|
 
 



### NOTES
![](https://raw.githubusercontent.com/notagenius/Action_Category_CVG/master/model_tips.jpg?token=AGSPNMZSFEZD4NICGXVOBZS57TVYU)

### Dataset Visualization
- TSNE

### Future Checklist:
Visulization
- [x] Confusion Matrix
- [x] Skeleton & Predicted label
- [x] Accuracy curve

Model Candidates
- [x] LSTM / GRU 25 frames * 2 sample rate 
- [ ] TCN Non Causal largest I can fit, it is fully conventional
- [x] TCN Causal 
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
