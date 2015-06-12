model step:

The model contains:

* map
* collection of environment objects
* collection of movers
* collection of weatherers
* a spills
* its own attributes


model loop
-----------
In pseudocode, the model loop is defined below. In the first step, it sets up the
model run and in subsequent steps the model moves and weathers elements. 

start at step_num = -1

if step_num == -1
    setup_model_run()

else:
    setup_time_step()
    move()
    weather()
    step_is_done()

step_num += 1

for sc in self.spills.items():
    num = sc.release_elements()     # initialize mover data arrays
    
    if num > 0:
        for w in weatherers:
            w.initialize_data()     # initialize weatherer data arrays

o = write_output()

return o

collection of environment objects
-----------------------------------
