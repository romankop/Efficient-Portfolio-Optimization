import pandas_datareader as wb
import pandas as pd
import matplotlib.pyplot as plot
import matplotlib
import datetime as date
import numpy as np
import openpyxl as excel
np.random.seed(71)

writer = pd.ExcelWriter('projectFinal.xlsx')

usa, rus = [], []
with open('inputReal.csv', 'r') as f:
    for row in f:
        if 'США' in row:
            usa.append(row.split(',')[0])
        elif 'РОССИЯ' in row:
            rus.append(row.split(',')[0])

start = date.datetime(2019, 1, 1)
end = date.datetime(2019, 12, 31)
tickers = wb.DataReader(usa, 'yahoo', start, end)['Close']

for comp in rus:
    data = wb.DataReader(comp, 'moex', start, end)
    data = data[~data['BOARDID'].isin(('RPUA', 'RPEU', 'RPEO'))]['CLOSE'] / 64
    tickers = tickers.merge(data, how='inner', left_index=True, right_index=True)

tickers.columns = usa + rus

tickers.to_excel(writer, sheet_name='InputData')

returns = tickers.pct_change().mean(skipna=True) * 253
returns.to_excel(writer, sheet_name='Annualized Returns')

covs = tickers.pct_change().cov() * 253
covs.to_excel(writer, sheet_name='Annualized Covariance')

weights = np.random.random(size=(100000, tickers.shape[1]))
weights = np.apply_along_axis(lambda x: x / x.sum(), 1, weights)

volatility = []
returns_weighted = []
covs = np.matrix(covs)

for weight in weights:
    return_per_weight = returns @ weight
    weight = weight.reshape((1,-1))
    vol = weight @ covs @ weight.T
    returns_weighted.append(return_per_weight)
    volatility.append(vol[0, 0])

risk_free = 0
sharpe = (np.array(returns_weighted) - risk_free) / np.sqrt(np.array(volatility))

ticker_columns = list(tickers.columns + ' wt')

df = pd.DataFrame(index=range(weights.shape[0]), columns=['Returns', 'Volatility', 'Sharpe Ratio'] + ticker_columns)

df[ticker_columns] = weights
df.Returns = returns_weighted
df.Volatility = np.sqrt(volatility)
df['Sharpe Ratio'] = sharpe

df.to_excel(writer, sheet_name='Portfolio Data')

max_sharpe = df.iloc[[df['Sharpe Ratio'].argmax()]]
max_sharpe.to_excel(writer, sheet_name='Title Sheet')

writer.save()

matplotlib.rc('font', size=8)
plot.style.use('ggplot')
plot.scatter(df.Volatility, df.Returns, s=1, c=df['Sharpe Ratio'], cmap='Greys')
plot.xlabel('Volatility (Std. Deviation)')
plot.ylabel('Expected Returns')
plot.colorbar(label='Sharpe Ratio')
plot.title('Efficient frontier n =' + str(weights.shape[0]))
plot.savefig('fig.png')

book = excel.load_workbook('projectFinal.xlsx')
sheet = book['Title Sheet']

img = excel.drawing.image.Image('fig.png')
img.anchor = 'B4'
sheet.add_image(img)

book.save('projectFinal.xlsx')

