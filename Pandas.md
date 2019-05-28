
# Pandas
Data structures in pandas
- `Series` objects: 1D array, similar to a column in a spreadsheet
- `DataFrame` objects: 2D table, similar to a spreadsheet
- `Panel` objects: Dictionary of DataFrames, similar to sheet in MS Excel

## Series


```python
import pandas as pd
import numpy as np

birthyear = pd.Series([1984, 1985, 1992])
print(birthyear)
print(birthyear.index)
print()

weight = pd.Series([68, 83, 112],index=["alice", "bob", "charles"])
print(weight)
print(weight.index)
```

    0    1984
    1    1985
    2    1992
    dtype: int64
    RangeIndex(start=0, stop=3, step=1)
    
    alice       68
    bob         83
    charles    112
    dtype: int64
    Index(['alice', 'bob', 'charles'], dtype='object')
    

# DataFrame


```python
weight = pd.Series([68, 83, 112],index=["alice", "bob", "charles"])
birthyear = pd.Series([1984, 1985, 1992], index=["bob", "alice", "charles"], name="year")
children = pd.Series([0, 3], index=["charles", "bob"])
hobby = pd.Series(["Biking", "Dancing"], index=["alice", "bob"])

people_dict = { "weight": weight,
                "birthyear": birthyear,
                "children": children,
                "hobby": hobby}

people = pd.DataFrame(people_dict)
print(people)
```

             birthyear  children    hobby  weight
    alice         1985       NaN   Biking      68
    bob           1984       3.0  Dancing      83
    charles       1992       0.0      NaN     112
    


```python
print('shape:', people.shape)
print(people.dtypes)
print('index:', people.index)
print('columns:', people.columns)
```

    shape: (3, 4)
    birthyear      int64
    children     float64
    hobby         object
    weight         int64
    dtype: object
    index: Index(['alice', 'bob', 'charles'], dtype='object')
    columns: Index(['birthyear', 'children', 'hobby', 'weight'], dtype='object')
    


```python
people['birthyear']
```




    alice      1985
    bob        1984
    charles    1992
    Name: birthyear, dtype: int64




```python
people['birthyear'] < 1990      # like numpy boolean array
```




    alice       True
    bob         True
    charles    False
    Name: birthyear, dtype: bool




```python
old_people = people[people['birthyear'] < 1990]    # like numpy boolean array indexing
old_people
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>birthyear</th>
      <th>children</th>
      <th>hobby</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>1985</td>
      <td>NaN</td>
      <td>Biking</td>
      <td>68</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>1984</td>
      <td>3.0</td>
      <td>Dancing</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>




```python
people_abbr = people[['birthyear', 'weight']]   # like numpy integer array indexing
people_abbr
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>birthyear</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alice</th>
      <td>1985</td>
      <td>68</td>
    </tr>
    <tr>
      <th>bob</th>
      <td>1984</td>
      <td>83</td>
    </tr>
    <tr>
      <th>charles</th>
      <td>1992</td>
      <td>112</td>
    </tr>
  </tbody>
</table>
</div>




```python
people['weight'].mean()
```




    87.66666666666667



# Example: data preparation

Read data

> Dataframes can also be easily exported and imported from CSV, Excel, JSON, HTML and SQL database. 


```python
url = 'http://bogotobogo.com/python/images/python_Pandas_NumPy_Matplotlib/HIP_star.dat'
df = pd.read_csv(url, sep='\s+')
print(df.shape)
df.head()
```

    (2720, 9)
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HIP</th>
      <th>Vmag</th>
      <th>RA</th>
      <th>DE</th>
      <th>Plx</th>
      <th>pmRA</th>
      <th>pmDE</th>
      <th>e_Plx</th>
      <th>B-V</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>9.27</td>
      <td>0.003797</td>
      <td>-19.498837</td>
      <td>21.90</td>
      <td>181.21</td>
      <td>-0.93</td>
      <td>3.10</td>
      <td>0.999</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38</td>
      <td>8.65</td>
      <td>0.111047</td>
      <td>-79.061831</td>
      <td>23.84</td>
      <td>162.30</td>
      <td>-62.40</td>
      <td>0.78</td>
      <td>0.778</td>
    </tr>
    <tr>
      <th>2</th>
      <td>47</td>
      <td>10.78</td>
      <td>0.135192</td>
      <td>-56.835248</td>
      <td>24.45</td>
      <td>-44.21</td>
      <td>-145.90</td>
      <td>1.97</td>
      <td>1.150</td>
    </tr>
    <tr>
      <th>3</th>
      <td>54</td>
      <td>10.57</td>
      <td>0.151656</td>
      <td>17.968956</td>
      <td>20.97</td>
      <td>367.14</td>
      <td>-19.49</td>
      <td>1.71</td>
      <td>1.030</td>
    </tr>
    <tr>
      <th>4</th>
      <td>74</td>
      <td>9.93</td>
      <td>0.221873</td>
      <td>35.752722</td>
      <td>24.22</td>
      <td>157.73</td>
      <td>-40.31</td>
      <td>1.36</td>
      <td>1.068</td>
    </tr>
  </tbody>
</table>
</div>



Data preprocessing: check data validity


```python
# check if a colum has no data (or NaN)
df.isnull().sum()
```




    HIP       0
    Vmag      1
    RA        1
    DE        1
    Plx       1
    pmRA      1
    pmDE      1
    e_Plx     1
    B-V      42
    dtype: int64




```python
# Drop any row if any of the column ha no data
df = df.dropna()
# Check again
df.isnull().sum()
```




    HIP      0
    Vmag     0
    RA       0
    DE       0
    Plx      0
    pmRA     0
    pmDE     0
    e_Plx    0
    B-V      0
    dtype: int64




```python
df.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Vmag</th>
      <th>RA</th>
      <th>DE</th>
      <th>Plx</th>
      <th>pmRA</th>
      <th>pmDE</th>
      <th>e_Plx</th>
      <th>B-V</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2678.000000</td>
      <td>2678.000000</td>
      <td>2678.000000</td>
      <td>2678.000000</td>
      <td>2678.000000</td>
      <td>2678.000000</td>
      <td>2678.000000</td>
      <td>2678.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8.214780</td>
      <td>173.528409</td>
      <td>-0.274356</td>
      <td>22.195403</td>
      <td>5.537058</td>
      <td>-63.534589</td>
      <td>1.544955</td>
      <td>0.761530</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.858407</td>
      <td>107.748388</td>
      <td>38.893512</td>
      <td>1.418260</td>
      <td>161.120941</td>
      <td>140.351882</td>
      <td>1.748178</td>
      <td>0.318188</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.450000</td>
      <td>0.003797</td>
      <td>-87.202730</td>
      <td>20.000000</td>
      <td>-868.010000</td>
      <td>-1392.300000</td>
      <td>0.450000</td>
      <td>-0.158000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.030000</td>
      <td>69.984258</td>
      <td>-31.800380</td>
      <td>20.980000</td>
      <td>-91.995000</td>
      <td>-129.967500</td>
      <td>0.870000</td>
      <td>0.560000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8.245000</td>
      <td>173.362326</td>
      <td>3.125766</td>
      <td>22.100000</td>
      <td>10.640000</td>
      <td>-48.680000</td>
      <td>1.130000</td>
      <td>0.710500</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.540000</td>
      <td>267.781761</td>
      <td>27.734524</td>
      <td>23.357500</td>
      <td>103.677500</td>
      <td>8.712500</td>
      <td>1.650000</td>
      <td>0.953000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>12.490000</td>
      <td>359.954685</td>
      <td>88.302681</td>
      <td>25.000000</td>
      <td>781.340000</td>
      <td>481.190000</td>
      <td>36.480000</td>
      <td>2.800000</td>
    </tr>
  </tbody>
</table>
</div>


