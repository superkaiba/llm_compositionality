from transformers import TrainerCallback, Trainer, TrainingArguments
import math

class SaveAtFractionalEpochCallback(TrainerCallback):
    def __init__(self, save_fractions, num_train_epochs, total_steps):
        # save_fractions is a list of fractions of epochs when you want to save the model (e.g., [0, 0.25, 0.5, 0.75])
        self.save_fractions = save_fractions
        self.num_train_epochs = num_train_epochs
        self.total_steps = total_steps
        self.steps_per_epoch = total_steps / num_train_epochs
        self.scheduled_save_steps = [
            math.floor(fraction * self.steps_per_epoch * num_train_epochs) for fraction in save_fractions
        ]
        print('self scheduled save steps: ', self.scheduled_save_steps)
    
    def on_step_end(self, args, state, control, **kwargs):
        # Save the model at steps corresponding to the specified fractions of epochs
        # print(state.global_step)
        if state.global_step in self.scheduled_save_steps:
            print(f"Saving model at step {state.global_step}, which corresponds to an epoch fraction")
            control.should_save = True
        else:
            # control.should_save = False
            pass


