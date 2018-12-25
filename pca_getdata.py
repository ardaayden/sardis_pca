import citiapi
import pandas as pd
import numpy as np
import datetime as dt
import math
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

today = dt.datetime.now()
one_year_before = today - dt.timedelta(days=730)
start_date = str(one_year_before.year)+str(one_year_before.month)+str(one_year_before.day)
end_date = str(today.year) + str(today.month) + str(today.day)

three_m = today - dt.timedelta(days=90)
three_m_int = three_m.year*10000 + three_m.month*100+three_m.day

access_token = citiapi.set_connection('2596265e-2930-46c8-95ac-f4503c7fb556',
                                      'rQ5bO6dJ6xT3dN5dL1oD4dN3cG0lL2hU3aG1jG1wN8fN2bW7qI')
client_id = '2596265e-2930-46c8-95ac-f4503c7fb556'

level = pd.DataFrame()
varswap = pd.DataFrame()
temp = citiapi.get_data(client_id, access_token, start_date, end_date, 'EQUITY.EQUITY_INDEX.92141.LEVEL.REUTERS')
temp['name'] = 'SPX'
level = level.append(temp)

temp = citiapi.get_data(client_id, access_token, start_date, end_date, 'EQUITY.EQUITY_INDEX.95393.LEVEL.REUTERS')
temp['name'] = 'SX7E'
level = level.append(temp)

level.to_csv('index_level.csv', index=False)


temp = citiapi.get_data(client_id, access_token, start_date, end_date, 'EQUITY.VARSWAP.SX7E.3M.EOD.CITI')
temp['name']= 'SX7E'
varswap = varswap.append(temp)

temp = citiapi.get_data(client_id, access_token, start_date, end_date, 'EQUITY.VARSWAP.SPX.3M.EOD.CITI')
temp['name']= 'SPX'
varswap = varswap.append(temp)
varswap.to_csv('index_varswap.csv', index=False)