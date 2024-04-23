import qafs
import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema
from pandera import io


fs = qafs.FeatureStore(
    project='test'
)

fs.create_namespace('example', description='Example Namespace', partition='date')

fs.create_feature(
    'numbers',
    namespace='example',
    description='Timeseries of numbers',
    check=Column(pa.Int, Check.greater_than(0))
)

# fs.create_features(
#     [
#         {
#             'name': 'numbers1',
#             'description': 'Timeseries of numbers',
#             'check': Column(pa.Int, Check.greater_than(0))
#         },
#         {
#             'name':'numbers2',
#             'description': 'Timeseries of numbers',
#             'check': Column(pa.Int, Check.greater_than(0))
#         }
#     ],
#     namespace='example'
# )

dts = pd.date_range('2020-01-01', '2021-02-09')
df = pd.DataFrame({'time': dts, 'numbers': list(range(1, len(dts) + 1))})

fs.save(df, index='time', features=['numbers'], namespace='example')

@fs.transform(
    'squared',
    t_namespace="example",
    from_features=['example/numbers'],
    check=Column(pa.Int, Check.greater_than(0))
)
def squared(df):
    return df ** 2

df_query = fs.load_features(
    ['example/numbers', 'example/squared'],
    from_date='2021-01-01',
    to_date='2021-01-31'
)
print(df_query.tail(1))