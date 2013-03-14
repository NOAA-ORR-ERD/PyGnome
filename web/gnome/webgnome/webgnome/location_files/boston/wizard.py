

def handle_input(model, data):
    if data.get('add_sewage_outfall', None) == 'yes':
        try:
            outfall_mover = model.movers.get('sewage_outfall')
        except KeyError:
            return

        # Turn off this mover - any better way?
        outfall_mover.active_stop = outfall_mover.active_start
