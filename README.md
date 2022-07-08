# Introduction
Within the framework of this project, a set of data on the state of health and features associated with stroke was analyzed. The main purpose of the analysis was to determine which values have the greatest impact on whether a person will have a stroke. The results also revealed differences between genders, place of residence, and working groups. In addition, 4 different machine learning models were created. Finally, these models were deployed in the telegram bot.

## Telegram bot
This bot consists of 8 different small bots that can give different forecasts. The bot runs on heroku, and it can be accessed at any time. The main code of the bot is in main.py. All used modules are located in separate folder.

**NOTE**: telegram bot is not fully optimized and can work quite slowly. The main problem is the way the data is collected.

### Using a telegram bot

Link - [Body healt prediction bot](https://t.me/body_health_predictions_bot)

The menu with all the bots could located in the lower left corner. To get a forecast, you need to answer all the questions. If you don't follow the rules, you will need to repeat the procedures again.  If you want to switch to another bot, you should write the /stop command and select another bot. 
If you choose another bot without stopping, the program will brake. This is also one of the aspects that needs to be updated later.

### Running on local pc

* Pip install requirements.txt in console
* Run main.py script

Enjoy

### Commands

/help - get guidance

/stroke - stroke prediction

/hypertension - prediction of hypertension

/glucose - predicting average glucose levels

/bmi - bmi prediction

/bmi_hypertension - prediction of hypertension and bmi

/bmi_glucose - prediction of average glucose level and bmi

/glucose_hypertension - prediction of average glucose levels and  hypertension

/all_three - prediction of average glucose level, hypertension and bmi
