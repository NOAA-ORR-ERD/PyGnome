import datetime


def handle_input(model, data):
    if data['use_custom_thing']:
        model.duration = datetime.timedelta(days=10)
