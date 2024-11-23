import numpy as np
import pandas as pd

# Test Dataframes and expected outputs

test_data = {
    "Name ": ["Tom", "Tom", "nick", "krish", "jack", "jill", "bill", "kim"],
    " Age!.": [20, 20, 21, 19, 18, 22, 23, 24],
    "OriGIn CountR,y": ["USA", "USA", "UK", "IND", "AUS", "USA", "UK", "IND"],
    "    SALARY ": [1000, 1000, 2000, 5000, np.nan, 6000, 8000, 7000],
    "sal.ary": [1000, 1000, 2000, 5000, np.nan, 6000, 8000, 7000],
    "Company": [
        "Google",
        "Google",
        "Microsoft",
        "Apple",
        "Amazon",
        "Google",
        "Microsoft",
        "Apple",
    ],
    "Time": [42943, 42943, 42944, np.nan, 42946, np.nan, 42948, 42949],
}

test_data_DROWS = {
    "name": ["Tom", "nick", "bill", "kim"],
    "age": [20, 21, 23, 24],
    "origin_country": ["USA", "UK", "UK", "IND"],
    "salary": [1000, 2000, 8000, 7000],
    "company": ["Google", "Microsoft", "Microsoft", "Apple"],
    "time": [42943, 42944, 42948, 42949],
}

test_data_DCOLS = {
    "name": ["Tom", "nick", "krish", "jack", "jill", "bill", "kim"],
    "age": [20, 21, 19, 18, 22, 23, 24],
    "origin_country": ["USA", "UK", "IND", "AUS", "USA", "UK", "IND"],
    "company": [
        "Google",
        "Microsoft",
        "Apple",
        "Amazon",
        "Google",
        "Microsoft",
        "Apple",
    ],
}

test_data_ZERO = {
    "name": ["Tom", "nick", "krish", "jack", "jill", "bill", "kim"],
    "age": [20, 21, 19, 18, 22, 23, 24],
    "origin_country": ["USA", "UK", "IND", "AUS", "USA", "UK", "IND"],
    "salary": [1000, 2000, 5000, 0, 6000, 8000, 7000],
    "company": [
        "Google",
        "Microsoft",
        "Apple",
        "Amazon",
        "Google",
        "Microsoft",
        "Apple",
    ],
    "time": [42943, 42944, 0, 42946, 0, 42948, 42949],
}

test_data_MEAN = {
    "name": ["Tom", "nick", "krish", "jack", "jill", "bill", "kim"],
    "age": [20, 21, 19, 18, 22, 23, 24],
    "origin_country": ["USA", "UK", "IND", "AUS", "USA", "UK", "IND"],
    "salary": [1000.0, 2000.0, 5000.0, 29000 / 6, 6000.0, 8000.0, 7000.0],
    "company": [
        "Google",
        "Microsoft",
        "Apple",
        "Amazon",
        "Google",
        "Microsoft",
        "Apple",
    ],
    "time": [42943.0, 42944.0, 42946.0, 42946.0, 42946.0, 42948.0, 42949.0],
}

test_data_MEDIAN = {
    "name": ["Tom", "nick", "krish", "jack", "jill", "bill", "kim"],
    "age": [20, 21, 19, 18, 22, 23, 24],
    "origin_country": ["USA", "UK", "IND", "AUS", "USA", "UK", "IND"],
    "salary": [1000.0, 2000.0, 5000.0, 5500.0, 6000.0, 8000.0, 7000.0],
    "company": [
        "Google",
        "Microsoft",
        "Apple",
        "Amazon",
        "Google",
        "Microsoft",
        "Apple",
    ],
    "time": [42943.0, 42944.0, 42946.0, 42946.0, 42946.0, 42948.0, 42949.0],
}

test_data_MODE = {
    "name": ["Tom", "nick", "krish", "jack", "jill", "bill", "kim"],
    "age": [20, 21, 19, 18, 22, 23, 24],
    "origin_country": ["USA", "UK", "IND", "AUS", "USA", "UK", "IND"],
    "salary": [1000.0, 2000.0, 5000.0, 1000.0, 6000.0, 8000.0, 7000.0],
    "company": [
        "Google",
        "Microsoft",
        "Apple",
        "Amazon",
        "Google",
        "Microsoft",
        "Apple",
    ],
    "time": [42943.0, 42944.0, 42943.0, 42946.0, 42943.0, 42948.0, 42949.0],
}

test_data_LERP = {
    "name": ["Tom", "nick", "krish", "jack", "jill", "bill", "kim"],
    "age": [20, 21, 19, 18, 22, 23, 24],
    "origin_country": ["USA", "UK", "IND", "AUS", "USA", "UK", "IND"],
    "salary": [1000, 2000, 5000, 5500, 6000, 8000, 7000],
    "company": [
        "Google",
        "Microsoft",
        "Apple",
        "Amazon",
        "Google",
        "Microsoft",
        "Apple",
    ],
    "time": [42943, 42944, 42945, 42946, 42947, 42948, 42949],
}

test_data_LOCF = {
    "name": ["Tom", "nick", "krish", "jack", "jill", "bill", "kim"],
    "age": [20, 21, 19, 18, 22, 23, 24],
    "origin_country": ["USA", "UK", "IND", "AUS", "USA", "UK", "IND"],
    "salary": [1000, 2000, 5000, 5000, 6000, 8000, 7000],
    "company": [
        "Google",
        "Microsoft",
        "Apple",
        "Amazon",
        "Google",
        "Microsoft",
        "Apple",
    ],
    "time": [42943, 42944, 42944, 42946, 42946, 42948, 42949],
}

test_data_NOCB = {
    "name": ["Tom", "nick", "krish", "jack", "jill", "bill", "kim"],
    "age": [20, 21, 19, 18, 22, 23, 24],
    "origin_country": ["USA", "UK", "IND", "AUS", "USA", "UK", "IND"],
    "salary": [1000, 2000, 5000, 6000, 6000, 8000, 7000],
    "company": [
        "Google",
        "Microsoft",
        "Apple",
        "Amazon",
        "Google",
        "Microsoft",
        "Apple",
    ],
    "time": [42943, 42944, 42946, 42946, 42948, 42948, 42949],
}

test_data_SAL_LERP_TIME_NOCB = {
    "name": ["Tom", "nick", "krish", "jack", "jill", "bill", "kim"],
    "age": [20, 21, 19, 18, 22, 23, 24],
    "origin_country": ["USA", "UK", "IND", "AUS", "USA", "UK", "IND"],
    "salary": [1000, 2000, 5000, 5500, 6000, 8000, 7000],
    "company": [
        "Google",
        "Microsoft",
        "Apple",
        "Amazon",
        "Google",
        "Microsoft",
        "Apple",
    ],
    "time": [42943, 42944, 42946, 42946, 42948, 42948, 42949],
}

test_data_SAL_LERP_TIME_DROP = {
    "name": ["Tom", "nick", "krish", "jack", "jill", "bill", "kim"],
    "age": [20, 21, 19, 18, 22, 23, 24],
    "origin_country": ["USA", "UK", "IND", "AUS", "USA", "UK", "IND"],
    "salary": [1000, 2000, 5000, 5500, 6000, 8000, 7000],
    "company": [
        "Google",
        "Microsoft",
        "Apple",
        "Amazon",
        "Google",
        "Microsoft",
        "Apple",
    ],
}

test_data_SAL_LERP_TIME_DROP_ROWS = {
    "name": ["Tom", "nick", "jack", "bill", "kim"],
    "age": [20, 21, 18, 23, 24],
    "origin_country": ["USA", "UK", "AUS", "UK", "IND"],
    "salary": [1000, 2000, 5500, 8000, 7000],
    "company": [
        "Google",
        "Microsoft",
        "Amazon",
        "Microsoft",
        "Apple",
    ],
    "time": [42943, 42944, 42946, 42948, 42949],
}

test_data_MEAN_INT = {
    "name": ["Tom", "nick", "krish", "jack", "jill", "bill", "kim"],
    "age": [20, 21, 19, 18, 22, 23, 24],
    "origin_country": ["USA", "UK", "IND", "AUS", "USA", "UK", "IND"],
    "salary": [1000, 2000, 5000, 4833, 6000, 8000, 7000],
    "company": [
        "Google",
        "Microsoft",
        "Apple",
        "Amazon",
        "Google",
        "Microsoft",
        "Apple",
    ],
    "time": [42943, 42944, 42946, 42946, 42946, 42948, 42949],
}

test_data_numeric_std = {
    "name": ["Tom", "nick", "bill", "kim"],
    "age": [
        -1.2649110640673518,
        -0.6324555320336759,
        0.6324555320336759,
        1.2649110640673518,
    ],
    "origin_country": ["USA", "UK", "UK", "IND"],
    "salary": [
        -1.150792911137501,
        -0.8219949365267865,
        1.150792911137501,
        0.8219949365267865,
    ],
    "company": [
        "Google",
        "Microsoft",
        "Microsoft",
        "Apple",
    ],
    "time": [
        -1.1766968108291043,
        -0.7844645405527362,
        0.7844645405527362,
        1.1766968108291043,
    ],
}

test_dataframe = pd.DataFrame(test_data)

test_string_data = {
    "name": ["John Doe34", "Jane_Doe   !", "4John Smith", " John Doe34"],
    "age": [25, 30, 35, 25],
    "city": ["  New York ", "Los Angeles~", "Chicago!", "New York"],
}

test_string_data_df = pd.DataFrame(test_string_data)
