import argparse
import random
import json
import traceback
from datetime import datetime
from collections import defaultdict

from tinydb import TinyDB, where
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Updater, CommandHandler, Filters, CallbackContext, MessageHandler, CallbackQueryHandler

sysrand = random.SystemRandom()


class Client:
    def __init__(self, token, db_path, input_path):
        self.db = TinyDB(db_path, ensure_ascii=False)
        self.input_path = input_path

        self.updater = Updater(token=token, use_context=True)
        self.updater.dispatcher.add_handler(CommandHandler("start", self.start, filters=Filters.command))
        self.updater.dispatcher.add_handler(CallbackQueryHandler(self.button))

        with open(input_path, "r") as r:
            self.records = [json.loads(line) for line in r]
            random.seed(8888)
            random.shuffle(self.records)

        self.last_records = defaultdict(None)
        self.chat2username = dict()
        print("Bot is ready!")

    def write_result(self, result, chat_id):
        if result == "skip":
            return True

        username = self.chat2username.get(chat_id)
        last_record = self.last_records.get(chat_id)
        if not last_record:
            return False

        last_record["label"] = result
        last_record["username"] = username
        last_record["chat_id"] = chat_id
        last_record["timestamp"] = int(datetime.now().timestamp())
        self.db.insert(last_record)
        return True

    def run(self):
        self.updater.start_polling()
        self.updater.idle()

    def start(self, update: Update, context: CallbackContext):
        self.show(update, context)

    def button(self, update: Update, context: CallbackContext) -> None:
        query = update.callback_query
        query.answer()

        data = query.data
        chat_id = update.effective_chat.id

        if self.write_result(data, chat_id):
            self.show(update, context)
        else:
            context.bot.send_message(text="Нужно перезапустить бот через '/start'", chat_id=chat_id)

    def sample_record(self, username, retries=300, max_overlap=3):
        found_new = False
        for _ in range(retries):
            record = sysrand.choice(self.records)
            instruction = record["instruction"]
            a_model = record["a_model"]
            b_model = record["b_model"]
            count = self.db.count((where("instruction") == instruction) & (where("a_model") == a_model) & (where("b_model") == b_model))
            if count >= max_overlap:
                continue
            if not self.db.contains((where("instruction") == instruction) & (where("username") == username) & (where("a_model") == a_model) & (where("b_model") == b_model)):
                found_new = True
                break
        if not found_new:
            print(f"No new tasks for {username}")
        return record

    def build_text(self, record):
        text = f"*Задание*: {record['instruction']}\n"
        if "input" in record and record["input"] and record["input"].strip() and record["input"].strip() != "<noinput>":
            text += f"*Вход*: {record['input']}\n"
        text += "\n\n"

        answer_a = record['a']
        if len(answer_a) > 1500:
            answer_a = answer_a[:1500] + "... (обрезано из-за Телеграма)"
        answer_b = record['b']
        if len(answer_b) > 1500:
            answer_b = answer_b[:1500] + "... (обрезано из-за Телеграма)"
        text += f"*Ответ A*:\n{answer_a}\n\n\n"
        text += f"*Ответ B*:\n{answer_b}"
        if len(text) > 4000:
            text = text[:4000] + "... (обрезано из-за Телеграма)"
        return text

    def show(self, update: Update, context: CallbackContext):
        chat_id = update.effective_chat.id
        if update.message:
            username = update.message.chat.username
            self.chat2username[chat_id] = username
        else:
            username = self.chat2username[chat_id]

        record = self.sample_record(username)
        self.last_records[chat_id] = record
        text = self.build_text(record)
        keyboard = [
            [
                InlineKeyboardButton("A лучше", callback_data="a"),
                InlineKeyboardButton("B лучше", callback_data="b")
            ],
            [
                InlineKeyboardButton("Одинаково", callback_data="equal")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        try:
            context.bot.send_message(
                text=text,
                reply_markup=reply_markup,
                parse_mode="Markdown",
                chat_id=chat_id
            )
        except Exception:
            print(traceback.format_exc())
            context.bot.send_message(
                text=text,
                reply_markup=reply_markup,
                chat_id=chat_id
            )


def main(
    token,
    input_path,
    db_path,
    seed
):
    client = Client(
        token=token,
        db_path=db_path,
        input_path=input_path
    )
    client.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--db-path", type=str, default="db.json")
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(**vars(args))
