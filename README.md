
## How to run the models
This repository contains multiple prediction models and some bandwidth dataset. 
To run any prediction models:
1. Open collab
2. Save the bandwidth data in csv, no header, in column, to Google Drive.
3. Manage the path, modify LSTM(x,y), where x is the prediction length, and y is the output length
4. Also tune parameters, and make predictions.

To run big change prediction model:
1. Run the big_change_prediction.ipynb in collab
2. After previous code finish, run the induced prediction.ipynb, which used information saved previously
