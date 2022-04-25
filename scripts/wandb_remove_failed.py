import os

import wandb
def remova_all_failed_runs(filters=None):
    if filters is None:
        filters = {
            "$or": [{"state":"killed"},{"state":"failed"},{"state":"crashed"}  ]
        }
    api = wandb.Api()
    remove_all = False
    for run in api.runs(
            path = f"{wandb.Api().default_entity}/kaggle-sorghum",
            filters=filters
    ):
        if remove_all:
            cmd = "Y"
        else:
            cmd = input(f"Do you want to remove {run} ? [Yes, No, All]")

        if cmd.lower() == "y" or cmd.lower() == "yes":
            run.delete()

        elif cmd.lower == "a" or cmd.lower() =="all":
            remove_all = True
            run.delete()
if __name__=="__main__":
    remova_all_failed_runs()
