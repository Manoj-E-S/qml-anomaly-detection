# CQU - Classical And Quantum ML

**cqu** is a classical and quantum machine learning library that can be used to quickly and easily reap benefits of both classical and quantum machine learning algorithms!

- Example usage
```
from cqu.preprocessing import Preprocessor, MissingValueStrategies
from cqu.quantum_embedding import QuantumClassifier

pp = Preprocessor("path/to/dataset")
strategies = { 
    'v3': MissingValueStrategies.FILL_MEDIAN, 
    'time': MissingValueStrategies.FILL_NOCB, 
    'class': MissingValueStrategies.DROP_ROWS 
}
pp.clean_missing(strategies)

dataset = pp.dataframe[['v3', 'v10', 'v11', 'class']]

qc = QuantumClassifier()
qc.train(dataset, "class")

test_df = pd.DataFrame(np.random.rand(5))
prediction = qc.predict(test_df).flatten()

if prediction[0] == 1:
    print("Prediction: Fraud")
else:
    print("Prediction: Non-fraud")
```