from cw2 import experiment


class MyIterativeExperiment(experiment.AbstractIterativeExperiment):
    # ...

    def initialize(self, config: dict, rep: int, logger) -> None:
        print("Ready to start repetition {}. Resetting everything.".format(rep))

    def iterate(self, config: dict, rep: int, n: int) -> dict:
        return {"Result": "Current Iteration is {}".format(n)}

    def save_state(self, config: dict, rep: int, n: int) -> None:
        if n % 50 == 0:
            print("I am stateless. Nothing to write to disk.")

    def finalize(self, surrender: bool = False, crash: bool = False):
        print("Finished. Closing Down.")


exp = AbstractIterativeExperiment()     # Initialize only global CONSTANTS
for r in repetitions:
    exp.initialize(...)    # Initialize / Reset the experiment for each repetition.
    for i in iterations:
        result = exp.iterate(...)   # Make a single iteration, return the result
        log(result)                 # Log the result
        exp.save_state(...)         # Save the current experiment state
    exp.finalize()      # Finalize / Clean the experiment after each repetition
