

def handle_input(model, data):
    if data.get('add_sewage_outfall', None) == 'yes':
        try:
            # TODO: Either we can specify movers by name like this, using a
            # custom ID, or we should enter the ID of the mover here.
            outfall_mover = model.movers.get('sewage_outfall')
        except KeyError:
            return

        outfall_mover.on = True
