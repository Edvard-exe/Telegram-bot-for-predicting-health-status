# Data loading
import logging
from typing import Tuple
import pandas as pd
import joblib

# Machine learning prediction
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImPipeline
from sklearn import model_selection

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImPipeline
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from lightgbm import LGBMClassifier

# Telegram bot api
from telegram import InlineKeyboardMarkup, InlineKeyboardButton, Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackQueryHandler,
    CallbackContext,
)


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


# State definitions for top level conversation
SELECTING_ACTION, ADDING_MEMBER, ADDING_SELF, DESCRIBING_SELF = map(chr, range(4))
# State definitions for second level conversation
(
    SELECTING_LEVEL,
    SELECTING_GENDER,
    RESIDENCE,
    SELECTING_WORK,
    SELECTING_MARIEGE,
    SELECTING_SMOKING,
) = map(chr, range(4, 10))
# State definitions for descriptions conversation
SELECTING_FEATURE, TYPING = map(chr, range(10, 12))
# Meta states
STOPPING, SHOWING = map(chr, range(12, 14))
# Shortcut for ConversationHandler.END
END = ConversationHandler.END

# Different constants
(
    ADD,
    GENDER,
    AGE,
    BMI,
    AVG,
    START_OVER,
    FEATURES,
    CURRENT_FEATURE,
    CURRENT_LEVEL,
    HYPER,
    HEART_DISEASE,
    STAT,
    CHILD,
    STROKE_STAT,
    HYPER_STAT,
    SMOKE_STAT,
    HOUSE,
    STROKE,
) = map(chr, range(14, 32))

df = pd.DataFrame()


def _name_switcher(level: str) -> Tuple[str, str]:
    if level == ADD:
        return "Male", "Female"


def _residence_switcher(level: str) -> Tuple[str, str]:
    return "Urban", "Rural"


def stroke(update: Update, context: CallbackContext) -> str:
    """Select an action: Add yourself/Show result."""

    text = (
        "You may choose to add a info about yourself or show the prediction results."
        " To abort, simply type /stop. Please follow all the rules, otherwise you will need to"
        " repeat every action."
    )

    buttons = [
        [
            InlineKeyboardButton(text="Yourself", callback_data=str(ADDING_MEMBER)),
        ],
        [
            InlineKeyboardButton(text="Show result", callback_data=str(SHOWING)),
        ],
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    if context.user_data.get(START_OVER):
        update.callback_query.answer()
        update.callback_query.edit_message_text(text=text, reply_markup=keyboard)
    else:
        update.message.reply_text(
            "Hi, I am a Stroke prediction bot and I am here to help you understand if you are at risk of stroke."
        )
        update.message.reply_text(text=text, reply_markup=keyboard)

    context.user_data[START_OVER] = False
    return SELECTING_ACTION


def hypertension(update: Update, context: CallbackContext) -> str:
    """Select an action: Add yourself/Show result."""
    text = (
        "You may choose to add a info about yourself or show the prediction results."
        " To abort, simply type /stop. Please follow all the rules, otherwise you will need to"
        "repeat every action."
    )

    buttons = [
        [
            InlineKeyboardButton(text="Yourself", callback_data=str(ADDING_MEMBER)),
        ],
        [
            InlineKeyboardButton(text="Show result", callback_data=str(SHOWING)),
        ],
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    if context.user_data.get(START_OVER):
        update.callback_query.answer()
        update.callback_query.edit_message_text(text=text, reply_markup=keyboard)
    else:
        update.message.reply_text(
            "Hi, I am a Hypertension prediction bot and I am here to help you understand if you are at risk of stroke."
        )
        update.message.reply_text(text=text, reply_markup=keyboard)

    context.user_data[START_OVER] = False

    return SELECTING_ACTION


def bmi(update: Update, context: CallbackContext) -> str:
    """Select an action: Add yourself/Show result."""
    text = (
        "You may choose to add a info about yourself or show the prediction results."
        " To abort, simply type /stop. Please follow all the rules, otherwise you will need to"
        " repeat every action."
    )

    buttons = [
        [
            InlineKeyboardButton(text="Yourself", callback_data=str(ADDING_MEMBER)),
        ],
        [
            InlineKeyboardButton(text="Show result", callback_data=str(SHOWING)),
        ],
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    if context.user_data.get(START_OVER):
        update.callback_query.answer()
        update.callback_query.edit_message_text(text=text, reply_markup=keyboard)
    else:
        update.message.reply_text(
            "Hi, I am a BMI prediction bot and I am here to predict your bmi."
        )
        update.message.reply_text(text=text, reply_markup=keyboard)

    context.user_data[START_OVER] = False
    return SELECTING_ACTION


def glucose(update: Update, context: CallbackContext) -> str:
    """Select an action: Add yourself/Show result."""

    text = (
        "You may choose to add a info about yourself or show the prediction results. "
        "To abort, simply type /stop. Please follow all the rules, otherwise you will need to"
        " repeat every action."
    )

    buttons = [
        [
            InlineKeyboardButton(text="Yourself", callback_data=str(ADDING_MEMBER)),
        ],
        [
            InlineKeyboardButton(text="Show result", callback_data=str(SHOWING)),
        ],
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    if context.user_data.get(START_OVER):
        update.callback_query.answer()
        update.callback_query.edit_message_text(text=text, reply_markup=keyboard)
    else:
        update.message.reply_text(
            "Hi, I am a Glucose prediction bot and I am here to predict your average glucose level."
        )
        update.message.reply_text(text=text, reply_markup=keyboard)

    context.user_data[START_OVER] = False
    return SELECTING_ACTION


def bmi_hypertension(update: Update, context: CallbackContext) -> str:
    """Select an action: Add yourself/Show result."""

    text = (
        "You may choose to add a info about yourself or show the prediction results. "
        "To abort, simply type /stop. Please follow all the rules, otherwise you will need to"
        " repeat every action."
    )

    buttons = [
        [
            InlineKeyboardButton(text="Yourself", callback_data=str(ADDING_MEMBER)),
        ],
        [
            InlineKeyboardButton(text="Show result", callback_data=str(SHOWING)),
        ],
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    if context.user_data.get(START_OVER):
        update.callback_query.answer()
        update.callback_query.edit_message_text(text=text, reply_markup=keyboard)
    else:
        update.message.reply_text(
            "Hi, I'm BMI and Hypertension prediction bot, and I'm here to predict these two parameters."
        )
        update.message.reply_text(text=text, reply_markup=keyboard)

    context.user_data[START_OVER] = False
    return SELECTING_ACTION


def bmi_glucose(update: Update, context: CallbackContext) -> str:
    """Select an action: Add yourself/Show result."""

    text = (
        "You may choose to add a info about yourself or show the prediction results. "
        "To abort, simply type /stop. Please follow all the rules, otherwise you will need to"
        " repeat every action."
    )

    buttons = [
        [
            InlineKeyboardButton(text="Yourself", callback_data=str(ADDING_MEMBER)),
        ],
        [
            InlineKeyboardButton(text="Show result", callback_data=str(SHOWING)),
        ],
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    if context.user_data.get(START_OVER):
        update.callback_query.answer()
        update.callback_query.edit_message_text(text=text, reply_markup=keyboard)
    else:
        update.message.reply_text(
            "Hi, I'm BMI and Average glucose level prediction bot, and I'm here to predict these two parameters."
        )
        update.message.reply_text(text=text, reply_markup=keyboard)

    context.user_data[START_OVER] = False
    return SELECTING_ACTION


def glucose_hypertension(update: Update, context: CallbackContext) -> str:
    """Select an action: Add yourself/Show result."""

    text = (
        "You may choose to add a info about yourself or show the prediction results. "
        "To abort, simply type /stop. Please follow all the rules, otherwise you will need to"
        " repeat every action."
    )

    buttons = [
        [
            InlineKeyboardButton(text="Yourself", callback_data=str(ADDING_MEMBER)),
        ],
        [
            InlineKeyboardButton(text="Show result", callback_data=str(SHOWING)),
        ],
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    if context.user_data.get(START_OVER):
        update.callback_query.answer()
        update.callback_query.edit_message_text(text=text, reply_markup=keyboard)
    else:
        update.message.reply_text(
            "Hi, I'm Average glucose level and Hypertension prediction bot, and I'm here to predict these two parameters."
        )
        update.message.reply_text(text=text, reply_markup=keyboard)

    context.user_data[START_OVER] = False
    return SELECTING_ACTION


def all_three(update: Update, context: CallbackContext) -> str:
    """Select an action: Add yourself/Show result."""

    text = (
        "You may choose to add a info about yourself or show the prediction results. "
        "To abort, simply type /stop. Please follow all the rules, otherwise you will need to"
        " repeat every action."
    )

    buttons = [
        [
            InlineKeyboardButton(text="Yourself", callback_data=str(ADDING_MEMBER)),
        ],
        [
            InlineKeyboardButton(text="Show result", callback_data=str(SHOWING)),
        ],
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    # If we're starting over we don't need to send a new message
    if context.user_data.get(START_OVER):
        update.callback_query.answer()
        update.callback_query.edit_message_text(text=text, reply_markup=keyboard)
    else:
        update.message.reply_text(
            "Hi, I'm Average glucose level, Hypertension and BMI prediction bot, and I'm here to predict these three parameters."
        )
        update.message.reply_text(text=text, reply_markup=keyboard)

    context.user_data[START_OVER] = False
    return SELECTING_ACTION


def show_data(update: Update, context: CallbackContext) -> str:
    """Returns prediction or rise an error"""

    def column():
        if "stroke" in df:
            return df.drop(columns=["stroke"])
        else:
            return df

    new_df = column()

    if len(new_df.columns) != 10:

        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text=f"There is no enough data", reply_markup=keyboard
        )
        user_data[START_OVER] = True

    elif not 0 <= new_df.iloc[0]["age"] <= 100:
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="You must specify the age within the specified limits. "
            "You will need to repeat everything again.",
            reply_markup=keyboard,
        )
        user_data[START_OVER] = True

    elif not 0 <= new_df.iloc[0]["bmi"] <= 100:
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="You must specify the bmi within the specified limits. "
            "You will need to repeat everything again.",
            reply_markup=keyboard,
        )
        user_data[START_OVER] = True

    elif not 50 <= new_df.iloc[0]["avg_glucose_level"] <= 290:
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="You must specify the glucose level within the specified limits. "
            "You will need to repeat everything again.",
            reply_markup=keyboard,
        )
        user_data[START_OVER] = True
    else:
        stroke_mod = pred_stroke()
        pipe_stroke = joblib.load("models/pipe.pkl")
        stroke_df = pipe_stroke.transform(new_df)
        stroke_res = stroke_mod.predict_proba(stroke_df)
        stroke_res = stroke_res[-1][-1]
        stroke_res = round(stroke_res, 2)
        stroke_res = stroke_res.item()
        risk = stroke_res
        if stroke_res >= 0.5:
            text = f"The probability of getting a stroke is {risk}. You are at risk!"
            user_data = context.user_data
            buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
            keyboard = InlineKeyboardMarkup(buttons)

            update.callback_query.answer()
            update.callback_query.edit_message_text(text=text, reply_markup=keyboard)
            user_data[START_OVER] = True
        else:
            text2 = f"The probability of getting a stroke is {stroke_res}. You are not at risk!"
            user_data = context.user_data
            buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
            keyboard = InlineKeyboardMarkup(buttons)

            update.callback_query.answer()
            update.callback_query.edit_message_text(text=text2, reply_markup=keyboard)
            user_data[START_OVER] = True

    return SHOWING


def show_data_hyper(update: Update, context: CallbackContext) -> str:
    """Returns prediction or rise an error"""

    def column():
        if "hypertension" in df:
            return df.drop(columns=["hypertension"])
        else:
            return df

    new_df = column()

    if len(new_df.columns) != 10:

        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="There is no enough data", reply_markup=keyboard
        )
        user_data[START_OVER] = True

    elif not 0 <= new_df.iloc[0]["age"] <= 100:
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="You must specify the age within the specified limits. "
            "You will need to repeat everything again.",
            reply_markup=keyboard,
        )
        user_data[START_OVER] = True

    elif not 0 <= new_df.iloc[0]["bmi"] <= 100:
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="You must specify the bmi within the specified limits. "
            "You will need to repeat everything again.",
            reply_markup=keyboard,
        )
        user_data[START_OVER] = True

    elif not 50 <= new_df.iloc[0]["avg_glucose_level"] <= 290:
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="You must specify the glucose level within the specified limits. "
            "You will need to repeat everything again.",
            reply_markup=keyboard,
        )
        user_data[START_OVER] = True
    else:
        hyper_mod, pipe_hyper = pred_hyper()
        hyper_df = pipe_hyper.transform(new_df)
        hyper_res = hyper_mod.predict_proba(hyper_df)
        hyper_res = hyper_res[-1][-1]
        hyper_res = round(hyper_res, 2)
        hyper_res = hyper_res.item()

        if hyper_res >= 0.5:
            text = f"The probability of getting a hypertension is {hyper_res}. You are at risk!"
            user_data = context.user_data
            buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
            keyboard = InlineKeyboardMarkup(buttons)

            update.callback_query.answer()
            update.callback_query.edit_message_text(text=text, reply_markup=keyboard)
            user_data[START_OVER] = True
        else:
            text2 = f"The probability of getting a hypertension is {hyper_res}. You are not at risk!"
            user_data = context.user_data
            buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
            keyboard = InlineKeyboardMarkup(buttons)

            update.callback_query.answer()
            update.callback_query.edit_message_text(text=text2, reply_markup=keyboard)
            user_data[START_OVER] = True

    return SHOWING


def show_data_bmi(update: Update, context: CallbackContext) -> str:
    """Returns prediction or rise an error"""

    def column():
        if "bmi" in df:
            return df.drop(columns=["bmi"])
        else:
            return df

    new_df = column()

    if len(new_df.columns) != 10:

        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="There is no enough data", reply_markup=keyboard
        )
        user_data[START_OVER] = True

    elif not 0 <= new_df.iloc[0]["age"] <= 100:
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="You must specify the age within the specified limits. "
            "You will need to repeat everything again.",
            reply_markup=keyboard,
        )
        user_data[START_OVER] = True

    elif not 50 <= new_df.iloc[0]["avg_glucose_level"] <= 290:
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="You must specify the glucose level within the specified limits. "
            "You will need to repeat everything again.",
            reply_markup=keyboard,
        )
        user_data[START_OVER] = True
    else:
        forest = joblib.load("models/forest_bmi_model.sav")
        forest = forest.predict(new_df)
        forest = forest[-1]
        text = f"The group of bmi you belong to: {forest} !"
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(text=text, reply_markup=keyboard)
        user_data[START_OVER] = True

    return SHOWING


def show_data_glucose(update: Update, context: CallbackContext) -> str:
    """Returns prediction or rise an error"""

    def column():
        if "avg_glucose_level" in df:
            return df.drop(columns=["avg_glucose_level"])
        else:
            return df

    new_df = column()

    if len(new_df.columns) != 10:

        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="There is no enough data", reply_markup=keyboard
        )
        user_data[START_OVER] = True

    elif not 0 <= new_df.iloc[0]["age"] <= 100:
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="You must specify the age within the specified limits. "
            "You will need to repeat everything again.",
            reply_markup=keyboard,
        )
        user_data[START_OVER] = True

    elif not 0 <= new_df.iloc[0]["bmi"] <= 100:
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)
        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="You must specify the bmi within the specified limits. "
            "You will need to repeat everything again.",
            reply_markup=keyboard,
        )
        user_data[START_OVER] = True

    else:
        glucose_mod = pred_glucose()
        glucose_mod = glucose_mod.predict(new_df)
        glucose_mod = glucose_mod[-1]

        text = f"The group of average glucose level you belong to: {glucose_mod} !"
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(text=text, reply_markup=keyboard)
        user_data[START_OVER] = True

    return SHOWING


def show_data_bh(update: Update, context: CallbackContext) -> str:
    """Returns prediction or rise an error"""

    if len(df.columns) != 11:

        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="There is no enough data", reply_markup=keyboard
        )
        user_data[START_OVER] = True

    elif not 0 <= df.iloc[0]["age"] <= 100:
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="You must specify the age within the specified limits. "
            "You will need to repeat everything again.",
            reply_markup=keyboard,
        )
        user_data[START_OVER] = True

    elif not 0 <= df.iloc[0]["bmi"] <= 100:
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="You must specify the bmi within the specified limits. "
            "You will need to repeat everything again.",
            reply_markup=keyboard,
        )
        user_data[START_OVER] = True

    elif not 50 <= df.iloc[0]["avg_glucose_level"] <= 290:
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="You must specify the glucose level within the specified limits. "
            "You will need to repeat everything again.",
            reply_markup=keyboard,
        )
        user_data[START_OVER] = True
    else:
        hyper_df = df.drop(columns=["hypertension"])
        bmi_df = df.drop(columns=["bmi"])
        forest = joblib.load("models/forest_bmi_model.sav")
        bmi_mod = forest.predict(bmi_df)
        bmi_mod = bmi_mod[-1]
        hyper_mod, pipe_hyper = pred_hyper()
        hyper_df = pipe_hyper.transform(hyper_df)
        hyper_res = hyper_mod.predict_proba(hyper_df)
        hyper_res = hyper_res[-1][-1]
        hyper_res = round(hyper_res, 2)
        hyper_res = hyper_res.item()

        if hyper_res >= 0.5:
            text = (
                f"The probability of getting a hypertension is {hyper_res}. "
                f"You are at risk! And the group of bmi you belong to: {bmi_mod}"
            )
            user_data = context.user_data
            buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
            keyboard = InlineKeyboardMarkup(buttons)

            update.callback_query.answer()
            update.callback_query.edit_message_text(text=text, reply_markup=keyboard)
            user_data[START_OVER] = True
        else:
            text2 = (
                f"The probability of getting a hypertension is {hyper_res}. You are not at risk! "
                f"And the group of bmi you belong to: {bmi_mod}"
            )
            user_data = context.user_data
            buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
            keyboard = InlineKeyboardMarkup(buttons)

            update.callback_query.answer()
            update.callback_query.edit_message_text(text=text2, reply_markup=keyboard)
            user_data[START_OVER] = True

    return SHOWING


def show_data_bg(update: Update, context: CallbackContext) -> str:
    """Returns prediction or rise an error"""

    if len(df.columns) != 11:

        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="There is no enough data", reply_markup=keyboard
        )
        user_data[START_OVER] = True

    elif not 0 <= df.iloc[0]["age"] <= 100:
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="You must specify the age within the specified limits. "
            "You will need to repeat everything again.",
            reply_markup=keyboard,
        )
        user_data[START_OVER] = True

    elif not 0 <= df.iloc[0]["bmi"] <= 100:
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="You must specify the bmi within the specified limits. "
            "You will need to repeat everything again.",
            reply_markup=keyboard,
        )
        user_data[START_OVER] = True

    elif not 50 <= df.iloc[0]["avg_glucose_level"] <= 290:
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="You must specify the glucose level within the specified limits. "
            "You will need to repeat everything again.",
            reply_markup=keyboard,
        )
        user_data[START_OVER] = True
    else:
        glucose_df = df.drop(columns=["avg_glucose_level"])
        bmi_df = df.drop(columns=["bmi"])
        forest = joblib.load("models/forest_bmi_model.sav")
        bmi_mod = forest.predict(bmi_df)
        bmi_mod = bmi_mod[-1]
        glucose_mod = pred_glucose()
        glucose_mod = glucose_mod.predict(glucose_df)
        glucose_mod = glucose_mod[-1]

        text = f"The group of average glucose level you belong to: {glucose_mod}. And the group of bmi you belong to: {bmi_mod}."
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(text=text, reply_markup=keyboard)
        user_data[START_OVER] = True

    return SHOWING


def show_data_gh(update: Update, context: CallbackContext) -> str:
    """Returns prediction or rise an error"""

    if len(df.columns) != 11:

        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="There is no enough data", reply_markup=keyboard
        )
        user_data[START_OVER] = True

    elif not 0 <= df.iloc[0]["age"] <= 100:
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="You must specify the age within the specified limits. "
            "You will need to repeat everything again.",
            reply_markup=keyboard,
        )
        user_data[START_OVER] = True

    elif not 0 <= df.iloc[0]["bmi"] <= 100:
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="You must specify the bmi within the specified limits. "
            "You will need to repeat everything again.",
            reply_markup=keyboard,
        )
        user_data[START_OVER] = True

    elif not 50 <= df.iloc[0]["avg_glucose_level"] <= 290:
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="You must specify the glucose level within the specified limits. "
            "You will need to repeat everything again.",
            reply_markup=keyboard,
        )
        user_data[START_OVER] = True
    else:
        hyper_df = df.drop(columns=["hypertension"])
        glucose_df = df.drop(columns=["avg_glucose_level"])
        glucose_mod = pred_glucose()
        glucose_mod = glucose_mod.predict(glucose_df)
        glucose_mod = glucose_mod[-1]
        hyper_mod, pipe_hyper = pred_hyper()
        hyper_df = pipe_hyper.transform(hyper_df)
        hyper_res = hyper_mod.predict_proba(hyper_df)
        hyper_res = hyper_res[-1][-1]
        hyper_res = round(hyper_res, 2)
        hyper_res = hyper_res.item()

        if hyper_res >= 0.5:
            text = (
                f"The probability of getting a hypertension is {hyper_res}. "
                f"You are at risk! And the group of Average glucose level you belong to: {glucose_mod}"
            )
            user_data = context.user_data
            buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
            keyboard = InlineKeyboardMarkup(buttons)

            update.callback_query.answer()
            update.callback_query.edit_message_text(text=text, reply_markup=keyboard)
            user_data[START_OVER] = True
        else:
            text2 = (
                f"The probability of getting a hypertension is {hyper_res}. You are not at risk! "
                f"And the group of Average glucose level you belong to: {glucose_mod}"
            )
            user_data = context.user_data
            buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
            keyboard = InlineKeyboardMarkup(buttons)

            update.callback_query.answer()
            update.callback_query.edit_message_text(text=text2, reply_markup=keyboard)
            user_data[START_OVER] = True

    return SHOWING


def show_data_three(update: Update, context: CallbackContext) -> str:
    """Returns prediction or rise an error"""

    if len(df.columns) != 11:

        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="There is no enough data", reply_markup=keyboard
        )
        user_data[START_OVER] = True

    elif not 0 <= df.iloc[0]["age"] <= 100:
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="You must specify the age within the specified limits. "
            "You will need to repeat everything again.",
            reply_markup=keyboard,
        )
        user_data[START_OVER] = True

    elif not 0 <= df.iloc[0]["bmi"] <= 100:
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="You must specify the bmi within the specified limits. "
            "You will need to repeat everything again.",
            reply_markup=keyboard,
        )
        user_data[START_OVER] = True

    elif not 50 <= df.iloc[0]["avg_glucose_level"] <= 290:
        user_data = context.user_data
        buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
        keyboard = InlineKeyboardMarkup(buttons)

        update.callback_query.answer()
        update.callback_query.edit_message_text(
            text="You must specify the glucose level within the specified limits. "
            "You will need to repeat everything again.",
            reply_markup=keyboard,
        )
        user_data[START_OVER] = True
    else:
        hyper_df = df.drop(columns=["hypertension"])
        glucose_df = df.drop(columns=["avg_glucose_level"])
        bmi_df = df.drop(columns=["bmi"])
        glucose_mod = pred_glucose()
        glucose_mod = glucose_mod.predict(glucose_df)
        glucose_mod = glucose_mod[-1]
        hyper_mod, pipe_hyper = pred_hyper()
        hyper_df = pipe_hyper.transform(hyper_df)
        hyper_res = hyper_mod.predict_proba(hyper_df)
        hyper_res = hyper_res[-1][-1]
        hyper_res = round(hyper_res, 2)
        hyper_res = hyper_res.item()
        forest = joblib.load("models/forest_bmi_model.sav")
        bmi_mod = forest.predict(bmi_df)
        bmi_mod = bmi_mod[-1]

        if hyper_res >= 0.5:
            text = (
                f"The probability of getting a hypertension is {hyper_res}. "
                f"You are at risk! The group of Average glucose level you belong to: {glucose_mod}. "
                f"Lastly the group of bmi you belong to: {bmi_mod}."
            )

            user_data = context.user_data
            buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
            keyboard = InlineKeyboardMarkup(buttons)

            update.callback_query.answer()
            update.callback_query.edit_message_text(text=text, reply_markup=keyboard)
            user_data[START_OVER] = True
        else:
            text2 = (
                f"The probability of getting a hypertension is {hyper_res}. You are not at risk! "
                f"The group of Average glucose level you belong to: {glucose_mod}. "
                f"Lastly the group of bmi you belong to: {bmi_mod}"
            )

            user_data = context.user_data
            buttons = [[InlineKeyboardButton(text="Back", callback_data=str(END))]]
            keyboard = InlineKeyboardMarkup(buttons)

            update.callback_query.answer()
            update.callback_query.edit_message_text(text=text2, reply_markup=keyboard)
            user_data[START_OVER] = True

    return SHOWING


def stop(update: Update, context: CallbackContext) -> int:
    """Stops conversation."""
    update.message.reply_text("Okay, bye.")

    return END


def end(update: Update, context: CallbackContext) -> int:
    """End conversation from InlineKeyboardButton."""
    update.message.reply_text("Okay, bye.")

    return END


# Second level conversation callbacks


def select_level(update: Update, context: CallbackContext) -> str:
    """Choose to add info or go back."""

    text = "You may add data about yourself. Also you can go back."
    buttons = [
        [
            InlineKeyboardButton(text="Add info", callback_data=str(ADD)),
        ],
        [
            InlineKeyboardButton(text="Back", callback_data=str(END)),
        ],
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    update.callback_query.answer()
    update.callback_query.edit_message_text(text=text, reply_markup=keyboard)

    return SELECTING_LEVEL


def select_gender(update: Update, context: CallbackContext) -> str:
    """Choose gender."""
    level = update.callback_query.data
    context.user_data[CURRENT_LEVEL] = level

    text = "Please choose, your gender."

    male, female = _name_switcher(level)

    buttons = [
        [
            InlineKeyboardButton(text=f"{male}", callback_data="Male"),
            InlineKeyboardButton(text=f"{female}", callback_data="Female"),
        ],
        [
            InlineKeyboardButton(text=f"Other", callback_data="Other"),
        ],
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    update.callback_query.answer()
    update.callback_query.edit_message_text(text=text, reply_markup=keyboard)

    return SELECTING_GENDER


def residence_type(update: Update, context: CallbackContext) -> str:
    """Choose residence type."""

    level = update.callback_query.data
    df["gender"] = [level]
    context.user_data[CURRENT_LEVEL] = level

    text = "Please choose, your residence type."

    buttons = [
        [
            InlineKeyboardButton(text=f"Urban", callback_data="Urban"),
            InlineKeyboardButton(text=f"Rural", callback_data="Rural"),
        ],
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    update.callback_query.answer()
    update.callback_query.edit_message_text(text=text, reply_markup=keyboard)
    context.user_data[FEATURES] = {HOUSE: update.callback_query.data}
    house_dict = context.user_data[FEATURES]

    return RESIDENCE


def work_type(update: Update, context: CallbackContext) -> str:
    """Choose work type."""

    level = update.callback_query.data
    df["Residence_type"] = [level]
    context.user_data[CURRENT_LEVEL] = level

    text = "Please choose, your work type"

    buttons = [
        [
            InlineKeyboardButton(text=f"Children", callback_data="children"),
            InlineKeyboardButton(text=f"Goverment", callback_data="Govt_job"),
            InlineKeyboardButton(text=f"Private", callback_data="Private"),
        ],
        [
            InlineKeyboardButton(text=f"Never worked", callback_data="Never_worked"),
            InlineKeyboardButton(
                text=f"Self - employed", callback_data="Self-employed"
            ),
        ],
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    update.callback_query.answer()
    update.callback_query.edit_message_text(text=text, reply_markup=keyboard)

    return SELECTING_WORK


def marriege_status(update: Update, context: CallbackContext) -> str:
    """Choose marriege status."""

    level = update.callback_query.data
    df["work_type"] = [level]
    text = "Are you married?"
    context.user_data[CURRENT_LEVEL] = level

    buttons = [
        [
            InlineKeyboardButton(text=f"Yes", callback_data="Yes"),
            InlineKeyboardButton(text=f"No", callback_data="No"),
        ],
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    update.callback_query.answer()
    update.callback_query.edit_message_text(text=text, reply_markup=keyboard)

    return SELECTING_MARIEGE


def smoking_status(update: Update, context: CallbackContext) -> str:
    """Choose smoking status."""

    level = update.callback_query.data
    df["ever_married"] = [level]
    context.user_data[CURRENT_LEVEL] = level

    text = "Please choose, your smoking status."

    buttons = [
        [
            InlineKeyboardButton(text=f"Formerly ", callback_data="formerly smoked"),
            InlineKeyboardButton(text=f"Never", callback_data="never smoked"),
        ],
        [
            InlineKeyboardButton(text=f"Smokes", callback_data="smokes"),
            InlineKeyboardButton(text=f"Unknown", callback_data="Unknown"),
        ],
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    update.callback_query.answer()
    update.callback_query.edit_message_text(text=text, reply_markup=keyboard)

    level = update.callback_query.data

    return SELECTING_SMOKING


def hyper_status(update: Update, context: CallbackContext) -> str:
    """Choose hypertension status."""

    level = update.callback_query.data
    df["smoking_status"] = [level]
    context.user_data[CURRENT_LEVEL] = level

    text = "Do you have hypertension?"

    buttons = [
        [
            InlineKeyboardButton(text=f"Yes", callback_data=1),
            InlineKeyboardButton(text=f"No", callback_data=0),
        ],
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    update.callback_query.answer()
    update.callback_query.edit_message_text(text=text, reply_markup=keyboard)

    return HYPER


def stroke_status(update: Update, context: CallbackContext) -> str:
    """Choose stroke status."""

    level = update.callback_query.data
    df["smoking_status"] = [level]
    context.user_data[CURRENT_LEVEL] = level

    text = "Did you have stroke ?"

    buttons = [
        [
            InlineKeyboardButton(text=f"Yes", callback_data=1),
            InlineKeyboardButton(text=f"No", callback_data=0),
        ],
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    update.callback_query.answer()
    update.callback_query.edit_message_text(text=text, reply_markup=keyboard)

    return STROKE


def info_smoke(update: Update, context: CallbackContext) -> str:
    """Saves smoking status."""

    level = update.callback_query.data
    df["smoking_status"] = [level]

    return SMOKE_STAT


def last_stroke(update: Update, context: CallbackContext) -> str:
    """Saves stroke status."""

    level = update.callback_query.data
    df["stroke"] = [int(level)]

    return STROKE_STAT


def hyper_stat(update: Update, context: CallbackContext) -> str:
    """Saves hypertension status."""

    level = update.callback_query.data
    df["hypertension"] = [int(level)]

    return HYPER_STAT


def heart_disease(update: Update, context: CallbackContext) -> str:
    """Choose heart disease status."""

    level = update.callback_query.data
    context.user_data[CURRENT_LEVEL] = level

    text = "Did you have heart disease?"

    buttons = [
        [
            InlineKeyboardButton(text=f"Yes", callback_data=1),
            InlineKeyboardButton(text=f"No", callback_data=0),
        ],
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    update.callback_query.answer()
    update.callback_query.edit_message_text(text=text, reply_markup=keyboard)

    return HEART_DISEASE


def last_stat(update: Update, context: CallbackContext) -> str:
    """Saves heart disease status."""

    level = update.callback_query.data
    df["heart_disease"] = [int(level)]

    return STAT


def end_second_level(update: Update, context: CallbackContext) -> int:
    """Return to top level conversation."""

    context.user_data[START_OVER] = True
    stroke(update, context)

    return END


def select_feature(update: Update, context: CallbackContext) -> str:
    """Select a feature to update for the person."""

    buttons = [
        [
            InlineKeyboardButton(text="BMI", callback_data="bmi"),
            InlineKeyboardButton(text="Age", callback_data="age"),
        ],
        [
            InlineKeyboardButton(text="Avg glucose", callback_data="avg_glucose_level"),
            InlineKeyboardButton(text="Done", callback_data=str(END)),
        ],
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    if not context.user_data.get(START_OVER):

        text = "Please select a feature to update."

        update.callback_query.answer()
        update.callback_query.edit_message_text(text=text, reply_markup=keyboard)

    else:
        text = "Got it! Please select a feature to update."
        update.message.reply_text(text=text, reply_markup=keyboard)

    context.user_data[START_OVER] = False
    return SELECTING_FEATURE


def select_featur_bmi(update: Update, context: CallbackContext) -> str:
    """Select a feature to update for the person."""
    buttons = [
        [
            InlineKeyboardButton(text="Avg glucose", callback_data="avg_glucose_level"),
            InlineKeyboardButton(text="Age", callback_data="age"),
        ],
        [
            InlineKeyboardButton(text="Done", callback_data=str(END)),
        ],
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    if not context.user_data.get(START_OVER):

        text = "Please select a feature to update."

        update.callback_query.answer()
        update.callback_query.edit_message_text(text=text, reply_markup=keyboard)

    else:
        text = "Got it! Please select a feature to update."
        update.message.reply_text(text=text, reply_markup=keyboard)

    context.user_data[START_OVER] = False
    return SELECTING_FEATURE


def select_featur_glucose(update: Update, context: CallbackContext) -> str:
    """Select a feature to update for the person."""
    buttons = [
        [
            InlineKeyboardButton(text="BMI", callback_data="bmi"),
            InlineKeyboardButton(text="Age", callback_data="age"),
        ],
        [
            InlineKeyboardButton(text="Done", callback_data=str(END)),
        ],
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    if not context.user_data.get(START_OVER):

        text = "Please select a feature to update."

        update.callback_query.answer()
        update.callback_query.edit_message_text(text=text, reply_markup=keyboard)

    else:
        text = "Got it! Please select a feature to update."
        update.message.reply_text(text=text, reply_markup=keyboard)

    context.user_data[START_OVER] = False

    return SELECTING_FEATURE


def ask_for_input(update: Update, context: CallbackContext) -> str:
    """Prompt user to input data for selected feature."""
    context.user_data[CURRENT_FEATURE] = update.callback_query.data

    info = "Please following values choose between this intervals"
    string1 = "• Select bmi between 0 and 100"
    string2 = "• Selct age between 0 and 100"
    string3 = "• Selct Avg glucose between 50 and 290"
    update.callback_query.answer()
    update.callback_query.edit_message_text(
        text=info + "\n" + string1 + "\n" + string2 + "\n" + string3
    )

    return TYPING


def ask_for_input_bmi(update: Update, context: CallbackContext) -> str:
    """Prompt user to input data for selected feature."""
    context.user_data[CURRENT_FEATURE] = update.callback_query.data

    info = "Please following values choose between this intervals"
    string2 = "• Selct age between 0 and 100"
    string3 = "• Selct Avg glucose between 50 and 290"
    update.callback_query.answer()
    update.callback_query.edit_message_text(text=info + "\n" + string2 + "\n" + string3)

    return TYPING


def ask_for_input_glucose(update: Update, context: CallbackContext) -> str:
    """Prompt user to input data for selected feature."""
    context.user_data[CURRENT_FEATURE] = update.callback_query.data

    info = "Please following values choose between this intervals"
    string2 = "• Selct age between 0 and 100"
    string3 = "• Select bmi between 0 and 100"
    update.callback_query.answer()
    update.callback_query.edit_message_text(text=info + "\n" + string2 + "\n" + string3)

    return TYPING


def save_input(update: Update, context: CallbackContext) -> str:
    """Save input for feature and return to feature selection."""
    user_data = context.user_data
    user_data[FEATURES][user_data[CURRENT_FEATURE]] = update.message.text
    user_data[START_OVER] = True

    return select_feature(update, context)


def save_input_bmi(update: Update, context: CallbackContext) -> str:
    """Save input for feature and return to feature selection."""
    user_data = context.user_data
    user_data[FEATURES][user_data[CURRENT_FEATURE]] = update.message.text
    user_data[START_OVER] = True

    return select_featur_bmi(update, context)


def save_input_glucose(update: Update, context: CallbackContext) -> str:
    """Save input for feature and return to feature selection."""
    user_data = context.user_data
    user_data[FEATURES][user_data[CURRENT_FEATURE]] = update.message.text
    user_data[START_OVER] = True

    return select_featur_glucose(update, context)


def end_describing(update: Update, context: CallbackContext) -> int:
    """End gathering of features and return to parent conversation."""
    user_data = context.user_data

    def recursive_lookup(k, d):
        if k in d:
            return d[k]
        for v in d.values():
            if isinstance(v, dict):
                return recursive_lookup(k, v)
        return None

    bmi = ("bmi", recursive_lookup("bmi", user_data))
    age = ("age", recursive_lookup("age", user_data))
    avg_glucose_level = (
        "avg_glucose_level",
        recursive_lookup("avg_glucose_level", user_data),
    )
    df["bmi"] = [float(bmi[-1])]
    df["age"] = [float(age[-1])]
    df["avg_glucose_level"] = [float(avg_glucose_level[-1])]

    level = user_data[CURRENT_LEVEL]
    if not user_data.get(level):
        user_data[level] = []
    user_data[level].append(user_data[FEATURES])

    select_level(update, context)

    return END


def end_describing_bmi(update: Update, context: CallbackContext) -> int:
    """End gathering of features and return to parent conversation."""
    user_data = context.user_data

    def recursive_lookup(k, d):
        if k in d:
            return d[k]
        for v in d.values():
            if isinstance(v, dict):
                return recursive_lookup(k, v)
        return None

    age = ("age", recursive_lookup("age", user_data))
    avg_glucose_level = (
        "avg_glucose_level",
        recursive_lookup("avg_glucose_level", user_data),
    )
    df["age"] = [float(age[-1])]
    df["avg_glucose_level"] = [float(avg_glucose_level[-1])]

    level = user_data[CURRENT_LEVEL]
    if not user_data.get(level):
        user_data[level] = []
    user_data[level].append(user_data[FEATURES])

    select_level(update, context)

    return END


def end_describing_glucose(update: Update, context: CallbackContext) -> int:
    """End gathering of features and return to parent conversation."""
    user_data = context.user_data

    def recursive_lookup(k, d):
        if k in d:
            return d[k]
        for v in d.values():
            if isinstance(v, dict):
                return recursive_lookup(k, v)
        return None

    age = ("age", recursive_lookup("age", user_data))
    bmi = ("bmi", recursive_lookup("bmi", user_data))
    df["age"] = [float(age[-1])]
    df["bmi"] = [float(bmi[-1])]

    level = user_data[CURRENT_LEVEL]
    if not user_data.get(level):
        user_data[level] = []
    user_data[level].append(user_data[FEATURES])

    select_level(update, context)

    return END


def stop_nested(update: Update, context: CallbackContext) -> str:
    """Completely end conversation from within nested conversation."""
    update.message.reply_text("Okay, bye.")

    return STOPPING


def help_command(update, context):
    """Help command."""
    update.message.reply_text(
        "You can find eight different bots in the menu in the lower-left corner. "
        "To get a forecast, you need to answer all the questions and complete all the steps. "
        "After you have passed all the necessary parameters, you can click 'Done' and return to "
        "the main menu to get the forecast. If you want to switch to another bot, you should write "
        "the /stop command and choose another bot. Also, I'm pretty slow, so if I haven't "
        "answered, make a few clicks on the right button. Good luck! "
    )


url = "https://res.cloudinary.com/dmc9xk3ka/raw/upload/v1649788826/healthcare-dataset-stroke-data_gmnbm8.csv"
stroke_df = pd.read_csv(url, index_col=0, sep=",").reset_index()


stroke_df = stroke_df.drop("id", 1)
stroke_df["bmi"] = stroke_df["bmi"].fillna(
    stroke_df.groupby("age")["bmi"].transform("mean")
)


def pred_stroke():

    stroke_mod = joblib.load("models/stroke_mod.pkl")

    return stroke_mod


def pred_hyper():

    X = stroke_df.drop(columns=["hypertension"])
    y = stroke_df["hypertension"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    pipe_hyper = joblib.load("models/hyper_pipe.pkl")
    X_train = pipe_hyper.fit_transform(X_train)
    hyper_mod = joblib.load("models/hyper_mod.pkl")

    return (hyper_mod, pipe_hyper)


def pred_glucose():
    mod_glucose = joblib.load("models/mod_glucose.pkl")

    return mod_glucose


def main() -> None:

    updater = Updater("5204857924:AAFIgJvQjoC4WCtKyMGVKeNolhptKbxudhM")

    dispatcher = updater.dispatcher

    # stroke bot conversation
    description_conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(select_feature)],
        states={
            SELECTING_FEATURE: [
                CallbackQueryHandler(ask_for_input, pattern="^(?!" + str(END) + ").*$")
            ],
            TYPING: [MessageHandler(Filters.text & ~Filters.command, save_input)],
        },
        fallbacks=[
            CallbackQueryHandler(end_describing, pattern="^" + str(END) + "$"),
            CommandHandler("stop", stop_nested),
        ],
        map_to_parent={
            END: SELECTING_LEVEL,
            STOPPING: STOPPING,
        },
    )

    add_member_conv = ConversationHandler(
        entry_points=[
            CallbackQueryHandler(select_level, pattern="^" + str(ADDING_MEMBER) + "$")
        ],
        states={
            SELECTING_LEVEL: [CallbackQueryHandler(select_gender, pattern=f"^{ADD}$")],
            SELECTING_GENDER: [CallbackQueryHandler(residence_type)],
            RESIDENCE: [CallbackQueryHandler(work_type)],
            SELECTING_WORK: [CallbackQueryHandler(marriege_status)],
            SELECTING_MARIEGE: [CallbackQueryHandler(smoking_status)],
            SELECTING_SMOKING: [CallbackQueryHandler(info_smoke)],
            SMOKE_STAT: [CallbackQueryHandler(hyper_status)],
            HYPER: [CallbackQueryHandler(hyper_stat)],
            HYPER_STAT: [CallbackQueryHandler(heart_disease)],
            HEART_DISEASE: [CallbackQueryHandler(last_stat)],
            STAT: [description_conv],
        },
        fallbacks=[
            CallbackQueryHandler(show_data, pattern="^" + str(SHOWING) + "$"),
            CallbackQueryHandler(end_second_level, pattern="^" + str(END) + "$"),
            CommandHandler("stop", stop_nested),
        ],
        map_to_parent={
            SHOWING: SHOWING,
            END: SELECTING_ACTION,
            STOPPING: END,
        },
    )

    selection_handlers = [
        add_member_conv,
        CallbackQueryHandler(show_data, pattern="^" + str(SHOWING) + "$"),
        CallbackQueryHandler(end, pattern="^" + str(END) + "$"),
    ]
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("stroke", stroke)],
        states={
            SHOWING: [CallbackQueryHandler(stroke, pattern="^" + str(END) + "$")],
            SELECTING_ACTION: selection_handlers,
            SELECTING_LEVEL: selection_handlers,
            DESCRIBING_SELF: [description_conv],
            STOPPING: [CommandHandler("stroke", stroke)],
        },
        fallbacks=[CommandHandler("stop", stop)],
    )

    # hypertension bot conversation
    description_hyper = ConversationHandler(
        entry_points=[CallbackQueryHandler(select_feature)],
        states={
            SELECTING_FEATURE: [
                CallbackQueryHandler(ask_for_input, pattern="^(?!" + str(END) + ").*$")
            ],
            TYPING: [MessageHandler(Filters.text & ~Filters.command, save_input)],
        },
        fallbacks=[
            CallbackQueryHandler(end_describing, pattern="^" + str(END) + "$"),
            CommandHandler("stop", stop_nested),
        ],
        map_to_parent={
            END: SELECTING_LEVEL,
            STOPPING: STOPPING,
        },
    )

    add_member_hypper = ConversationHandler(
        entry_points=[
            CallbackQueryHandler(select_level, pattern="^" + str(ADDING_MEMBER) + "$")
        ],
        states={
            SELECTING_LEVEL: [CallbackQueryHandler(select_gender, pattern=f"^{ADD}$")],
            SELECTING_GENDER: [CallbackQueryHandler(residence_type)],
            RESIDENCE: [CallbackQueryHandler(work_type)],
            SELECTING_WORK: [CallbackQueryHandler(marriege_status)],
            SELECTING_MARIEGE: [CallbackQueryHandler(smoking_status)],
            SELECTING_SMOKING: [CallbackQueryHandler(stroke_status)],
            STROKE: [CallbackQueryHandler(last_stroke)],
            STROKE_STAT: [CallbackQueryHandler(heart_disease)],
            HEART_DISEASE: [CallbackQueryHandler(last_stat)],
            STAT: [description_conv],
        },
        fallbacks=[
            CallbackQueryHandler(show_data_hyper, pattern="^" + str(SHOWING) + "$"),
            CallbackQueryHandler(end_second_level, pattern="^" + str(END) + "$"),
            CommandHandler("stop", stop_nested),
        ],
        map_to_parent={
            SHOWING: SHOWING,
            END: SELECTING_ACTION,
            STOPPING: END,
        },
    )

    selection_hyper = [
        add_member_hypper,
        CallbackQueryHandler(show_data_hyper, pattern="^" + str(SHOWING) + "$"),
        CallbackQueryHandler(end, pattern="^" + str(END) + "$"),
    ]
    conv_hyper = ConversationHandler(
        entry_points=[CommandHandler("hypertension", hypertension)],
        states={
            SHOWING: [CallbackQueryHandler(hypertension, pattern="^" + str(END) + "$")],
            SELECTING_ACTION: selection_hyper,
            SELECTING_LEVEL: selection_hyper,
            DESCRIBING_SELF: [description_hyper],
            STOPPING: [CommandHandler("stroke", stroke)],
        },
        fallbacks=[CommandHandler("stop", stop)],
    )

    # bmi bot conversation
    description_bmi = ConversationHandler(
        entry_points=[CallbackQueryHandler(select_featur_bmi)],
        states={
            SELECTING_FEATURE: [
                CallbackQueryHandler(
                    ask_for_input_bmi, pattern="^(?!" + str(END) + ").*$"
                )
            ],
            TYPING: [MessageHandler(Filters.text & ~Filters.command, save_input_bmi)],
        },
        fallbacks=[
            CallbackQueryHandler(end_describing_bmi, pattern="^" + str(END) + "$"),
            CommandHandler("stop", stop_nested),
        ],
        map_to_parent={
            END: SELECTING_LEVEL,
            STOPPING: STOPPING,
        },
    )

    add_member_bmi = ConversationHandler(
        entry_points=[
            CallbackQueryHandler(select_level, pattern="^" + str(ADDING_MEMBER) + "$")
        ],
        states={
            SELECTING_LEVEL: [CallbackQueryHandler(select_gender, pattern=f"^{ADD}$")],
            SELECTING_GENDER: [CallbackQueryHandler(residence_type)],
            RESIDENCE: [CallbackQueryHandler(work_type)],
            SELECTING_WORK: [CallbackQueryHandler(marriege_status)],
            SELECTING_MARIEGE: [CallbackQueryHandler(smoking_status)],
            SELECTING_SMOKING: [CallbackQueryHandler(info_smoke)],
            SMOKE_STAT: [CallbackQueryHandler(hyper_status)],
            HYPER: [CallbackQueryHandler(hyper_stat)],
            HYPER_STAT: [CallbackQueryHandler(stroke_status)],
            STROKE: [CallbackQueryHandler(last_stroke)],
            STROKE_STAT: [CallbackQueryHandler(heart_disease)],
            HEART_DISEASE: [CallbackQueryHandler(last_stat)],
            STAT: [description_bmi],
        },
        fallbacks=[
            CallbackQueryHandler(show_data_bmi, pattern="^" + str(SHOWING) + "$"),
            CallbackQueryHandler(end_second_level, pattern="^" + str(END) + "$"),
            CommandHandler("stop", stop_nested),
        ],
        map_to_parent={
            SHOWING: SHOWING,
            END: SELECTING_ACTION,
            STOPPING: END,
        },
    )

    selection_bmi = [
        add_member_bmi,
        CallbackQueryHandler(show_data_bmi, pattern="^" + str(SHOWING) + "$"),
        CallbackQueryHandler(end, pattern="^" + str(END) + "$"),
    ]
    conv_bmi = ConversationHandler(
        entry_points=[CommandHandler("bmi", bmi)],
        states={
            SHOWING: [CallbackQueryHandler(bmi, pattern="^" + str(END) + "$")],
            SELECTING_ACTION: selection_bmi,
            SELECTING_LEVEL: selection_bmi,
            DESCRIBING_SELF: [description_bmi],
            STOPPING: [CommandHandler("bmi", bmi)],
        },
        fallbacks=[CommandHandler("stop", stop)],
    )

    # glucose bot conversation
    description_glucose = ConversationHandler(
        entry_points=[CallbackQueryHandler(select_featur_glucose)],
        states={
            SELECTING_FEATURE: [
                CallbackQueryHandler(
                    ask_for_input_glucose, pattern="^(?!" + str(END) + ").*$"
                )
            ],
            TYPING: [
                MessageHandler(Filters.text & ~Filters.command, save_input_glucose)
            ],
        },
        fallbacks=[
            CallbackQueryHandler(end_describing_glucose, pattern="^" + str(END) + "$"),
            CommandHandler("stop", stop_nested),
        ],
        map_to_parent={
            END: SELECTING_LEVEL,
            STOPPING: STOPPING,
        },
    )

    add_member_glucose = ConversationHandler(
        entry_points=[
            CallbackQueryHandler(select_level, pattern="^" + str(ADDING_MEMBER) + "$")
        ],
        states={
            SELECTING_LEVEL: [CallbackQueryHandler(select_gender, pattern=f"^{ADD}$")],
            SELECTING_GENDER: [CallbackQueryHandler(residence_type)],
            RESIDENCE: [CallbackQueryHandler(work_type)],
            SELECTING_WORK: [CallbackQueryHandler(marriege_status)],
            SELECTING_MARIEGE: [CallbackQueryHandler(smoking_status)],
            SELECTING_SMOKING: [CallbackQueryHandler(info_smoke)],
            SMOKE_STAT: [CallbackQueryHandler(hyper_status)],
            HYPER: [CallbackQueryHandler(hyper_stat)],
            HYPER_STAT: [CallbackQueryHandler(stroke_status)],
            STROKE: [CallbackQueryHandler(last_stroke)],
            STROKE_STAT: [CallbackQueryHandler(heart_disease)],
            HEART_DISEASE: [CallbackQueryHandler(last_stat)],
            STAT: [description_glucose],
        },
        fallbacks=[
            CallbackQueryHandler(show_data_glucose, pattern="^" + str(SHOWING) + "$"),
            CallbackQueryHandler(end_second_level, pattern="^" + str(END) + "$"),
            CommandHandler("stop", stop_nested),
        ],
        map_to_parent={
            SHOWING: SHOWING,
            END: SELECTING_ACTION,
            STOPPING: END,
        },
    )

    selection_glucose = [
        add_member_glucose,
        CallbackQueryHandler(show_data_glucose, pattern="^" + str(SHOWING) + "$"),
        CallbackQueryHandler(end, pattern="^" + str(END) + "$"),
    ]
    conv_glucose = ConversationHandler(
        entry_points=[CommandHandler("glucose", glucose)],
        states={
            SHOWING: [CallbackQueryHandler(glucose, pattern="^" + str(END) + "$")],
            SELECTING_ACTION: selection_glucose,
            SELECTING_LEVEL: selection_glucose,
            DESCRIBING_SELF: [description_glucose],
            STOPPING: [CommandHandler("glucose", glucose)],
        },
        fallbacks=[CommandHandler("stop", stop)],
    )

    # bmi and hypertension bot conversation
    description_bh = ConversationHandler(
        entry_points=[CallbackQueryHandler(select_feature)],
        states={
            SELECTING_FEATURE: [
                CallbackQueryHandler(ask_for_input, pattern="^(?!" + str(END) + ").*$")
            ],
            TYPING: [MessageHandler(Filters.text & ~Filters.command, save_input)],
        },
        fallbacks=[
            CallbackQueryHandler(end_describing, pattern="^" + str(END) + "$"),
            CommandHandler("stop", stop_nested),
        ],
        map_to_parent={
            END: SELECTING_LEVEL,
            STOPPING: STOPPING,
        },
    )

    add_member_bh = ConversationHandler(
        entry_points=[
            CallbackQueryHandler(select_level, pattern="^" + str(ADDING_MEMBER) + "$")
        ],
        states={
            SELECTING_LEVEL: [CallbackQueryHandler(select_gender, pattern=f"^{ADD}$")],
            SELECTING_GENDER: [CallbackQueryHandler(residence_type)],
            RESIDENCE: [CallbackQueryHandler(work_type)],
            SELECTING_WORK: [CallbackQueryHandler(marriege_status)],
            SELECTING_MARIEGE: [CallbackQueryHandler(smoking_status)],
            SELECTING_SMOKING: [CallbackQueryHandler(info_smoke)],
            SMOKE_STAT: [CallbackQueryHandler(hyper_status)],
            HYPER: [CallbackQueryHandler(hyper_stat)],
            HYPER_STAT: [CallbackQueryHandler(stroke_status)],
            STROKE: [CallbackQueryHandler(last_stroke)],
            STROKE_STAT: [CallbackQueryHandler(heart_disease)],
            HEART_DISEASE: [CallbackQueryHandler(last_stat)],
            STAT: [description_bh],
        },
        fallbacks=[
            CallbackQueryHandler(show_data_bh, pattern="^" + str(SHOWING) + "$"),
            CallbackQueryHandler(end_second_level, pattern="^" + str(END) + "$"),
            CommandHandler("stop", stop_nested),
        ],
        map_to_parent={
            SHOWING: SHOWING,
            END: SELECTING_ACTION,
            STOPPING: END,
        },
    )

    selection_bh = [
        add_member_bh,
        CallbackQueryHandler(show_data_bh, pattern="^" + str(SHOWING) + "$"),
        CallbackQueryHandler(end, pattern="^" + str(END) + "$"),
    ]
    conv_bh = ConversationHandler(
        entry_points=[CommandHandler("bmi_hypertension", bmi_hypertension)],
        states={
            SHOWING: [
                CallbackQueryHandler(bmi_hypertension, pattern="^" + str(END) + "$")
            ],
            SELECTING_ACTION: selection_bh,
            SELECTING_LEVEL: selection_bh,
            DESCRIBING_SELF: [description_bh],
            STOPPING: [CommandHandler("bmi_hypertension", bmi_hypertension)],
        },
        fallbacks=[CommandHandler("stop", stop)],
    )

    # bmi and glucose bot conversation
    description_bg = ConversationHandler(
        entry_points=[CallbackQueryHandler(select_feature)],
        states={
            SELECTING_FEATURE: [
                CallbackQueryHandler(ask_for_input, pattern="^(?!" + str(END) + ").*$")
            ],
            TYPING: [MessageHandler(Filters.text & ~Filters.command, save_input)],
        },
        fallbacks=[
            CallbackQueryHandler(end_describing, pattern="^" + str(END) + "$"),
            CommandHandler("stop", stop_nested),
        ],
        map_to_parent={
            END: SELECTING_LEVEL,
            STOPPING: STOPPING,
        },
    )

    add_member_bg = ConversationHandler(
        entry_points=[
            CallbackQueryHandler(select_level, pattern="^" + str(ADDING_MEMBER) + "$")
        ],
        states={
            SELECTING_LEVEL: [CallbackQueryHandler(select_gender, pattern=f"^{ADD}$")],
            SELECTING_GENDER: [CallbackQueryHandler(residence_type)],
            RESIDENCE: [CallbackQueryHandler(work_type)],
            SELECTING_WORK: [CallbackQueryHandler(marriege_status)],
            SELECTING_MARIEGE: [CallbackQueryHandler(smoking_status)],
            SELECTING_SMOKING: [CallbackQueryHandler(info_smoke)],
            SMOKE_STAT: [CallbackQueryHandler(hyper_status)],
            HYPER: [CallbackQueryHandler(hyper_stat)],
            HYPER_STAT: [CallbackQueryHandler(stroke_status)],
            STROKE: [CallbackQueryHandler(last_stroke)],
            STROKE_STAT: [CallbackQueryHandler(heart_disease)],
            HEART_DISEASE: [CallbackQueryHandler(last_stat)],
            STAT: [description_bh],
        },
        fallbacks=[
            CallbackQueryHandler(show_data_bg, pattern="^" + str(SHOWING) + "$"),
            CallbackQueryHandler(end_second_level, pattern="^" + str(END) + "$"),
            CommandHandler("stop", stop_nested),
        ],
        map_to_parent={
            SHOWING: SHOWING,
            END: SELECTING_ACTION,
            STOPPING: END,
        },
    )

    selection_bg = [
        add_member_bg,
        CallbackQueryHandler(show_data_bg, pattern="^" + str(SHOWING) + "$"),
        CallbackQueryHandler(end, pattern="^" + str(END) + "$"),
    ]
    conv_bg = ConversationHandler(
        entry_points=[CommandHandler("bmi_glucose", bmi_glucose)],
        states={
            SHOWING: [CallbackQueryHandler(bmi_glucose, pattern="^" + str(END) + "$")],
            SELECTING_ACTION: selection_bg,
            SELECTING_LEVEL: selection_bg,
            DESCRIBING_SELF: [description_bg],
            STOPPING: [CommandHandler("bmi_glucose", bmi_glucose)],
        },
        fallbacks=[CommandHandler("stop", stop)],
    )

    description_gh = ConversationHandler(
        entry_points=[CallbackQueryHandler(select_feature)],
        states={
            SELECTING_FEATURE: [
                CallbackQueryHandler(ask_for_input, pattern="^(?!" + str(END) + ").*$")
            ],
            TYPING: [MessageHandler(Filters.text & ~Filters.command, save_input)],
        },
        fallbacks=[
            CallbackQueryHandler(end_describing, pattern="^" + str(END) + "$"),
            CommandHandler("stop", stop_nested),
        ],
        map_to_parent={
            END: SELECTING_LEVEL,
            STOPPING: STOPPING,
        },
    )

    # hypertension and glucose bot conversation
    add_member_gh = ConversationHandler(
        entry_points=[
            CallbackQueryHandler(select_level, pattern="^" + str(ADDING_MEMBER) + "$")
        ],
        states={
            SELECTING_LEVEL: [CallbackQueryHandler(select_gender, pattern=f"^{ADD}$")],
            SELECTING_GENDER: [CallbackQueryHandler(residence_type)],
            RESIDENCE: [CallbackQueryHandler(work_type)],
            SELECTING_WORK: [CallbackQueryHandler(marriege_status)],
            SELECTING_MARIEGE: [CallbackQueryHandler(smoking_status)],
            SELECTING_SMOKING: [CallbackQueryHandler(info_smoke)],
            SMOKE_STAT: [CallbackQueryHandler(hyper_status)],
            HYPER: [CallbackQueryHandler(hyper_stat)],
            HYPER_STAT: [CallbackQueryHandler(stroke_status)],
            STROKE: [CallbackQueryHandler(last_stroke)],
            STROKE_STAT: [CallbackQueryHandler(heart_disease)],
            HEART_DISEASE: [CallbackQueryHandler(last_stat)],
            STAT: [description_bh],
        },
        fallbacks=[
            CallbackQueryHandler(show_data_gh, pattern="^" + str(SHOWING) + "$"),
            CallbackQueryHandler(end_second_level, pattern="^" + str(END) + "$"),
            CommandHandler("stop", stop_nested),
        ],
        map_to_parent={
            SHOWING: SHOWING,
            END: SELECTING_ACTION,
            STOPPING: END,
        },
    )

    selection_gh = [
        add_member_gh,
        CallbackQueryHandler(show_data_gh, pattern="^" + str(SHOWING) + "$"),
        CallbackQueryHandler(end, pattern="^" + str(END) + "$"),
    ]
    conv_gh = ConversationHandler(
        entry_points=[CommandHandler("glucose_hypertension", glucose_hypertension)],
        states={
            SHOWING: [
                CallbackQueryHandler(glucose_hypertension, pattern="^" + str(END) + "$")
            ],
            SELECTING_ACTION: selection_gh,
            SELECTING_LEVEL: selection_gh,
            DESCRIBING_SELF: [description_gh],
            STOPPING: [CommandHandler("glucose_hypertension", glucose_hypertension)],
        },
        fallbacks=[CommandHandler("stop", stop)],
    )

    description_three = ConversationHandler(
        entry_points=[CallbackQueryHandler(select_feature)],
        states={
            SELECTING_FEATURE: [
                CallbackQueryHandler(ask_for_input, pattern="^(?!" + str(END) + ").*$")
            ],
            TYPING: [MessageHandler(Filters.text & ~Filters.command, save_input)],
        },
        fallbacks=[
            CallbackQueryHandler(end_describing, pattern="^" + str(END) + "$"),
            CommandHandler("stop", stop_nested),
        ],
        map_to_parent={
            END: SELECTING_LEVEL,
            STOPPING: STOPPING,
        },
    )

    # all three values bot conversation
    add_member_three = ConversationHandler(
        entry_points=[
            CallbackQueryHandler(select_level, pattern="^" + str(ADDING_MEMBER) + "$")
        ],
        states={
            SELECTING_LEVEL: [CallbackQueryHandler(select_gender, pattern=f"^{ADD}$")],
            SELECTING_GENDER: [CallbackQueryHandler(residence_type)],
            RESIDENCE: [CallbackQueryHandler(work_type)],
            SELECTING_WORK: [CallbackQueryHandler(marriege_status)],
            SELECTING_MARIEGE: [CallbackQueryHandler(smoking_status)],
            SELECTING_SMOKING: [CallbackQueryHandler(info_smoke)],
            SMOKE_STAT: [CallbackQueryHandler(hyper_status)],
            HYPER: [CallbackQueryHandler(hyper_stat)],
            HYPER_STAT: [CallbackQueryHandler(stroke_status)],
            STROKE: [CallbackQueryHandler(last_stroke)],
            STROKE_STAT: [CallbackQueryHandler(heart_disease)],
            HEART_DISEASE: [CallbackQueryHandler(last_stat)],
            STAT: [description_bh],
        },
        fallbacks=[
            CallbackQueryHandler(show_data_gh, pattern="^" + str(SHOWING) + "$"),
            CallbackQueryHandler(end_second_level, pattern="^" + str(END) + "$"),
            CommandHandler("stop", stop_nested),
        ],
        map_to_parent={
            SHOWING: SHOWING,
            END: SELECTING_ACTION,
            STOPPING: END,
        },
    )

    selection_three = [
        add_member_three,
        CallbackQueryHandler(show_data_three, pattern="^" + str(SHOWING) + "$"),
        CallbackQueryHandler(end, pattern="^" + str(END) + "$"),
    ]
    conv_three = ConversationHandler(
        entry_points=[CommandHandler("all_three", all_three)],
        states={
            SHOWING: [CallbackQueryHandler(all_three, pattern="^" + str(END) + "$")],
            SELECTING_ACTION: selection_three,
            SELECTING_LEVEL: selection_three,
            DESCRIBING_SELF: [description_three],
            STOPPING: [CommandHandler("all_three", all_three)],
        },
        fallbacks=[CommandHandler("stop", stop)],
    )

    dispatcher.add_handler(conv_handler)
    dispatcher.add_handler(conv_hyper)
    dispatcher.add_handler(conv_bmi)
    dispatcher.add_handler(conv_glucose)
    dispatcher.add_handler(conv_bh)
    dispatcher.add_handler(conv_bg)
    dispatcher.add_handler(conv_gh)
    dispatcher.add_handler(conv_three)
    dispatcher.add_handler(CommandHandler("help", help_command))

    updater.start_polling()

    updater.idle()


if __name__ == "__main__":
    main()
