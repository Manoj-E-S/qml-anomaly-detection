import pandas as pd

test_data = {
    "Name ": ["Tom", "nick", "krish", "jack"],
    " Age!.": [20, 21, 19, 18],
    "OriGIn CountR,y": ["USA", "UK", "IND", "AUS"],
    "    SALARY ": [1000, 2000, "IDK", 4000],
}

test_dataframe = pd.DataFrame(test_data)
