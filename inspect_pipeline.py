import pandas as pd
from pathlib import Path
root = Path('data/processed/pipeline_outputs')
for fn in ['X_train.csv','X_test.csv']:
    p = root / fn
    print(f'\nFILE: {p}')
    if p.exists():
        df = pd.read_csv(p, nrows=5)
        print('columns:', list(df.columns))
        print('dtypes:')
        print(df.dtypes.value_counts())
        print(df.head(3).to_dict(orient='list'))
    else:
        print('missing')
